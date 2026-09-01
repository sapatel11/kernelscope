from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from kernelscope import residual_rmsnorm_cuda, residual_rmsnorm_cuda_naive


SHAPE = (8, 128, 768)
EPSILON = 1.0e-5
WARMUP = 10
PROFILE_ITERATIONS = 10


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling")

    torch.manual_seed(2026)
    device = torch.device("cuda")
    dtype = torch.float16

    input_tensor = torch.randn(SHAPE, device=device, dtype=dtype)
    residual = torch.randn_like(input_tensor)
    weight = torch.randn(SHAPE[-1], device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(WARMUP):
            residual_rmsnorm_cuda_naive(input_tensor, residual, weight, EPSILON)
            residual_rmsnorm_cuda(input_tensor, residual, weight, EPSILON)
        torch.cuda.synchronize()

        for _ in range(PROFILE_ITERATIONS):
            residual_rmsnorm_cuda_naive(input_tensor, residual, weight, EPSILON)
        torch.cuda.synchronize()

        for _ in range(PROFILE_ITERATIONS):
            residual_rmsnorm_cuda(input_tensor, residual, weight, EPSILON)
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
