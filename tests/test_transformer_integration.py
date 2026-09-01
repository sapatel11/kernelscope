import pytest
import torch

from kernelscope import KernelScopeFeedForwardBlock, KernelScopeMiniTransformer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for transformer integration tests",
)


def _make_block(hidden_size: int = 128, intermediate_size: int = 256):
    torch.manual_seed(2026)
    block = KernelScopeFeedForwardBlock(hidden_size, intermediate_size).cuda().half()
    block.eval()
    return block


def test_feed_forward_block_matches_eager_path() -> None:
    block = _make_block()
    torch.manual_seed(2027)
    input_tensor = torch.randn((2, 16, 128), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(input_tensor)

    with torch.inference_mode():
        eager_branch, eager_residual = block.forward_eager(input_tensor, residual)
        optimized_branch, optimized_residual = block.forward_optimized(input_tensor, residual)

    torch.testing.assert_close(
        optimized_residual,
        eager_residual,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        optimized_branch,
        eager_branch,
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_mini_transformer_matches_eager_path() -> None:
    torch.manual_seed(2026)
    model = KernelScopeMiniTransformer(
        hidden_size=128,
        intermediate_size=256,
        layers=4,
    ).cuda().half().eval()

    torch.manual_seed(2028)
    input_tensor = torch.randn((1, 32, 128), device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        eager = model.forward_eager(input_tensor)
        optimized = model.forward_optimized(input_tensor)

    torch.testing.assert_close(
        optimized,
        eager,
        rtol=3.0e-3,
        atol=3.0e-3,
    )


def test_transformer_output_shape_matches_input() -> None:
    model = KernelScopeMiniTransformer(
        hidden_size=64,
        intermediate_size=128,
        layers=2,
    ).cuda().half().eval()
    input_tensor = torch.randn((2, 8, 64), device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        output = model.forward_optimized(input_tensor)

    assert output.shape == input_tensor.shape
    assert output.dtype == torch.float16
    assert output.device.type == "cuda"
