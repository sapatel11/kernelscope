from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

# Running this file directly sets sys.path[0] to the benchmarks directory.
# Add the repository root so the local kernelscope package is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from kernelscope import residual_rmsnorm_cuda_naive, residual_rmsnorm_reference


SHAPE = (8, 128, 768)
DTYPE = torch.float16
EPSILON = 1.0e-5
WARMUP_ITERATIONS = 50
MEASURED_ITERATIONS = 200


def _nvcc_version() -> str:
    try:
        completed = subprocess.run(
            ["nvcc", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else "unavailable"


def _time_cuda_events(operation, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()

    return start.elapsed_time(end) / iterations


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.manual_seed(2026)
    device = torch.device("cuda")

    input_tensor = torch.randn(SHAPE, device=device, dtype=DTYPE)
    residual = torch.randn(SHAPE, device=device, dtype=DTYPE)
    weight = torch.randn(SHAPE[-1], device=device, dtype=DTYPE)

    eager = lambda: residual_rmsnorm_reference(
        input_tensor,
        residual,
        weight,
        EPSILON,
    )
    naive = lambda: residual_rmsnorm_cuda_naive(
        input_tensor,
        residual,
        weight,
        EPSILON,
    )

    # Force CUDA context initialization before warmup and timing.
    eager()
    naive()
    torch.cuda.synchronize()

    for _ in range(WARMUP_ITERATIONS):
        eager()
        naive()
    torch.cuda.synchronize()

    eager_ms = _time_cuda_events(eager, MEASURED_ITERATIONS)
    naive_ms = _time_cuda_events(naive, MEASURED_ITERATIONS)

    expected = eager()
    actual = naive()
    torch.cuda.synchronize()
    max_abs_error = (actual.float() - expected.float()).abs().max().item()

    props = torch.cuda.get_device_properties(device)
    ratio = eager_ms / naive_ms if naive_ms > 0 else float("inf")

    print("KernelScope Stage 2 — Naive CUDA residual + RMSNorm")
    print("=" * 60)
    print(f"GPU: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"NVCC: {_nvcc_version()}")
    print(f"Shape: {SHAPE}")
    print(f"dtype: {DTYPE}")
    print(f"epsilon: {EPSILON}")
    print(f"warmup iterations: {WARMUP_ITERATIONS}")
    print(f"measured iterations: {MEASURED_ITERATIONS}")
    print()
    print(f"PyTorch eager formula: {eager_ms:.6f} ms / call")
    print(f"Naive CUDA operator:   {naive_ms:.6f} ms / call")
    print(f"Eager / naive ratio:   {ratio:.3f}x")
    print(f"Maximum abs error:     {max_abs_error:.8g}")
    print()
    print("These are measurements from the machine that executed this script.")
    print("The naive CUDA implementation is a correctness baseline, not an optimized kernel.")


if __name__ == "__main__":
    main()
