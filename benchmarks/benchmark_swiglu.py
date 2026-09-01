from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from kernelscope import swiglu_cuda, swiglu_reference


SHAPE = (8, 128, 2048)
DTYPE = torch.float16
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
    gate = torch.randn(SHAPE, device=device, dtype=DTYPE)
    value = torch.randn(SHAPE, device=device, dtype=DTYPE)

    eager = lambda: swiglu_reference(gate, value)
    fused = lambda: swiglu_cuda(gate, value)

    eager()
    fused()
    torch.cuda.synchronize()

    for _ in range(WARMUP_ITERATIONS):
        eager()
        fused()
    torch.cuda.synchronize()

    eager_ms = _time_cuda_events(eager, MEASURED_ITERATIONS)
    fused_ms = _time_cuda_events(fused, MEASURED_ITERATIONS)

    expected = eager()
    actual = fused()
    torch.cuda.synchronize()
    max_abs_error = (actual.float() - expected.float()).abs().max().item()

    props = torch.cuda.get_device_properties(device)
    speedup = eager_ms / fused_ms if fused_ms > 0 else float("inf")

    print("KernelScope Stage 4 — Fused SwiGLU")
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
    print(f"warmup iterations: {WARMUP_ITERATIONS}")
    print(f"measured iterations: {MEASURED_ITERATIONS}")
    print()
    print(f"PyTorch eager SwiGLU: {eager_ms:.6f} ms / call")
    print(f"Fused CUDA SwiGLU:    {fused_ms:.6f} ms / call")
    print(f"Eager / fused:        {speedup:.3f}x")
    print(f"Maximum abs error:    {max_abs_error:.8g}")
    print()
    print("All performance values are measurements from the machine that executed this script.")


if __name__ == "__main__":
    main()
