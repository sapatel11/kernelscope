from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from kernelscope import (
    residual_rmsnorm_cuda,
    residual_rmsnorm_cuda_naive,
    residual_rmsnorm_reference,
)


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

    eager = lambda: residual_rmsnorm_reference(input_tensor, residual, weight, EPSILON)
    naive = lambda: residual_rmsnorm_cuda_naive(input_tensor, residual, weight, EPSILON)
    optimized = lambda: residual_rmsnorm_cuda(input_tensor, residual, weight, EPSILON)

    eager()
    naive()
    optimized()
    torch.cuda.synchronize()

    for _ in range(WARMUP_ITERATIONS):
        eager()
        naive()
        optimized()
    torch.cuda.synchronize()

    eager_ms = _time_cuda_events(eager, MEASURED_ITERATIONS)
    naive_ms = _time_cuda_events(naive, MEASURED_ITERATIONS)
    optimized_ms = _time_cuda_events(optimized, MEASURED_ITERATIONS)

    expected = eager()
    naive_output = naive()
    optimized_output = optimized()
    torch.cuda.synchronize()

    naive_error = (naive_output.float() - expected.float()).abs().max().item()
    optimized_error = (optimized_output.float() - expected.float()).abs().max().item()

    props = torch.cuda.get_device_properties(device)
    print("KernelScope Stage 3 — Optimized residual + RMSNorm")
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
    print(f"Optimized CUDA:        {optimized_ms:.6f} ms / call")
    print(f"Eager / optimized:     {eager_ms / optimized_ms:.3f}x")
    print(f"Naive / optimized:     {naive_ms / optimized_ms:.3f}x")
    print(f"Naive max abs error:   {naive_error:.8g}")
    print(f"Optimized max error:   {optimized_error:.8g}")
    print()
    print("All performance values are measurements from the machine that executed this script.")


if __name__ == "__main__":
    main()
