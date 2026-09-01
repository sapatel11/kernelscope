from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from kernelscope import KernelScopeMiniTransformer


BATCH = 8
SEQUENCE = 128
HIDDEN = 768
INTERMEDIATE = 2048
LAYERS = 4
WARMUP = 10
PROFILE_ITERATIONS = 10


def _run_path(model: KernelScopeMiniTransformer, input_tensor: torch.Tensor, *, optimized: bool) -> None:
    for _ in range(PROFILE_ITERATIONS):
        if optimized:
            model.forward_optimized(input_tensor)
        else:
            model.forward_eager(input_tensor)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling")

    torch.manual_seed(2026)
    device = torch.device("cuda")
    dtype = torch.float16

    model = KernelScopeMiniTransformer(
        HIDDEN,
        INTERMEDIATE,
        layers=LAYERS,
    ).to(device=device, dtype=dtype).eval()

    input_tensor = torch.randn(
        (BATCH, SEQUENCE, HIDDEN),
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        for _ in range(WARMUP):
            model.forward_eager(input_tensor)
            model.forward_optimized(input_tensor)
        torch.cuda.synchronize()

        torch.cuda.nvtx.range_push("kernelscope_eager_model")
        _run_path(model, input_tensor, optimized=False)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("kernelscope_optimized_model")
        _run_path(model, input_tensor, optimized=True)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()
