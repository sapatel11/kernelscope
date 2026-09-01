from __future__ import annotations

import platform
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from kernelscope import (
    residual_rmsnorm_cuda,
    residual_rmsnorm_cuda_naive,
    residual_rmsnorm_cuda_vectorized,
    residual_rmsnorm_reference,
)


SHAPES = [
    (1, 128, 512),
    (1, 512, 768),
    (8, 128, 768),
    (16, 256, 1024),
]
WARMUP = 30
ITERATIONS = 100
ROUNDS = 5
EPSILON = 1.0e-5


def time_round(operation) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERATIONS):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / ITERATIONS


def median_latency(operation) -> tuple[float, list[float]]:
    for _ in range(WARMUP):
        operation()
    torch.cuda.synchronize()
    rounds = [time_round(operation) for _ in range(ROUNDS)]
    return statistics.median(rounds), rounds


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    props = torch.cuda.get_device_properties(0)
    print("KernelScope Stage 9 — RMSNorm vectorization experiment")
    print("=" * 72)
    print(f"GPU: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"warmup iterations: {WARMUP}")
    print(f"iterations per round: {ITERATIONS}")
    print(f"rounds: {ROUNDS}")
    print()

    for shape in SHAPES:
        torch.manual_seed(2030)
        x = torch.randn(shape, device="cuda", dtype=torch.float16)
        residual = torch.randn_like(x)
        weight = torch.randn(shape[-1], device="cuda", dtype=torch.float16)

        naive = lambda: residual_rmsnorm_cuda_naive(x, residual, weight, EPSILON)
        scalar = lambda: residual_rmsnorm_cuda(x, residual, weight, EPSILON)
        vectorized = lambda: residual_rmsnorm_cuda_vectorized(x, residual, weight, EPSILON)

        naive_ms, _ = median_latency(naive)
        scalar_ms, scalar_rounds = median_latency(scalar)
        vector_ms, vector_rounds = median_latency(vectorized)

        expected = residual_rmsnorm_reference(x, residual, weight, EPSILON)
        actual = vectorized()
        torch.cuda.synchronize()
        max_error = (actual.float() - expected.float()).abs().max().item()

        print(
            f"shape={shape}: naive={naive_ms:.6f} ms, "
            f"warp={scalar_ms:.6f} ms, half2={vector_ms:.6f} ms, "
            f"warp/half2={scalar_ms / vector_ms:.3f}x, "
            f"naive/half2={naive_ms / vector_ms:.3f}x, max_err={max_error:.8g}"
        )
        print(f"  warp rounds:  {[round(v, 6) for v in scalar_rounds]}")
        print(f"  half2 rounds: {[round(v, 6) for v in vector_rounds]}")

    print()
    print("Use the median across rounds as the comparison metric; do not assume half2 wins.")


if __name__ == "__main__":
    main()
