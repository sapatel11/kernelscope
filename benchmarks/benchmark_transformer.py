from __future__ import annotations

import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from kernelscope import KernelScopeFeedForwardBlock, KernelScopeMiniTransformer


BATCH = 8
SEQUENCE = 128
HIDDEN = 768
INTERMEDIATE = 2048
LAYERS = 4
WARMUP_ITERATIONS = 30
MEASURED_ITERATIONS = 100


def _time_cuda(operation, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _warmup(*operations) -> None:
    for _ in range(WARMUP_ITERATIONS):
        for operation in operations:
            operation()
    torch.cuda.synchronize()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.manual_seed(2026)
    device = torch.device("cuda")
    dtype = torch.float16

    block = KernelScopeFeedForwardBlock(HIDDEN, INTERMEDIATE).to(device=device, dtype=dtype).eval()
    model = KernelScopeMiniTransformer(HIDDEN, INTERMEDIATE, layers=LAYERS).to(
        device=device,
        dtype=dtype,
    ).eval()

    input_tensor = torch.randn(
        (BATCH, SEQUENCE, HIDDEN),
        device=device,
        dtype=dtype,
    )
    residual = torch.randn_like(input_tensor)

    with torch.inference_mode():
        eager_block = lambda: block.forward_eager(input_tensor, residual)
        optimized_block = lambda: block.forward_optimized(input_tensor, residual)
        eager_model = lambda: model.forward_eager(input_tensor)
        optimized_model = lambda: model.forward_optimized(input_tensor)

        eager_block()
        optimized_block()
        eager_model()
        optimized_model()
        torch.cuda.synchronize()

        _warmup(eager_block, optimized_block, eager_model, optimized_model)

        eager_block_ms = _time_cuda(eager_block, MEASURED_ITERATIONS)
        optimized_block_ms = _time_cuda(optimized_block, MEASURED_ITERATIONS)
        eager_model_ms = _time_cuda(eager_model, MEASURED_ITERATIONS)
        optimized_model_ms = _time_cuda(optimized_model, MEASURED_ITERATIONS)

        eager_output = eager_model()
        optimized_output = optimized_model()
        torch.cuda.synchronize()

    max_abs_error = (
        optimized_output.float() - eager_output.float()
    ).abs().max().item()

    tokens = BATCH * SEQUENCE
    eager_tokens_per_second = tokens / (eager_model_ms / 1000.0)
    optimized_tokens_per_second = tokens / (optimized_model_ms / 1000.0)

    props = torch.cuda.get_device_properties(device)
    block_speedup = eager_block_ms / optimized_block_ms
    model_speedup = eager_model_ms / optimized_model_ms

    print("KernelScope Stage 5 — Mini transformer integration")
    print("=" * 60)
    print(f"GPU: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"Shape: ({BATCH}, {SEQUENCE}, {HIDDEN})")
    print(f"Intermediate size: {INTERMEDIATE}")
    print(f"Layers: {LAYERS}")
    print(f"dtype: {dtype}")
    print(f"warmup iterations: {WARMUP_ITERATIONS}")
    print(f"measured iterations: {MEASURED_ITERATIONS}")
    print()
    print(f"Eager block:           {eager_block_ms:.6f} ms")
    print(f"Optimized block:       {optimized_block_ms:.6f} ms")
    print(f"Block speedup:         {block_speedup:.3f}x")
    print()
    print(f"Eager 4-layer model:   {eager_model_ms:.6f} ms")
    print(f"Optimized 4-layer:     {optimized_model_ms:.6f} ms")
    print(f"Model speedup:         {model_speedup:.3f}x")
    print(f"Eager throughput:      {eager_tokens_per_second:,.1f} tokens/s")
    print(f"Optimized throughput:  {optimized_tokens_per_second:,.1f} tokens/s")
    print(f"Maximum abs error:     {max_abs_error:.8g}")
    print()
    print("Linear projections are shared PyTorch nn.Linear layers backed by CUDA/cuBLAS.")
    print("Only residual RMSNorm and SwiGLU differ between eager and optimized paths.")


if __name__ == "__main__":
    main()
