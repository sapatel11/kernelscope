import pytest
import torch

from kernelscope import (
    residual_rmsnorm_cuda,
    residual_rmsnorm_cuda_vectorized,
    residual_rmsnorm_reference,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDA robustness tests",
)


@pytest.mark.parametrize("hidden", [1, 3, 7, 31, 255, 513, 1025])
def test_vectorized_path_falls_back_safely_for_odd_hidden_sizes(hidden: int) -> None:
    torch.manual_seed(2026)
    shape = (2, 5, hidden)
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float16)
    expected = residual_rmsnorm_reference(x, residual, weight, 1.0e-5)
    actual = residual_rmsnorm_cuda_vectorized(x, residual, weight, 1.0e-5)
    torch.testing.assert_close(actual, expected, rtol=1.0e-3, atol=1.0e-3)


@pytest.mark.parametrize("hidden", [2, 8, 64, 512, 768, 1024, 2048])
def test_vectorized_path_matches_scalar_optimized(hidden: int) -> None:
    torch.manual_seed(2027)
    shape = (4, 17, hidden)
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float16)
    scalar = residual_rmsnorm_cuda(x, residual, weight, 1.0e-5)
    vectorized = residual_rmsnorm_cuda_vectorized(x, residual, weight, 1.0e-5)
    torch.testing.assert_close(vectorized, scalar, rtol=1.0e-3, atol=1.0e-3)


def test_repeated_runs_are_deterministic() -> None:
    torch.manual_seed(2028)
    x = torch.randn((8, 128, 768), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(768, device="cuda", dtype=torch.float16)
    first = residual_rmsnorm_cuda_vectorized(x, residual, weight, 1.0e-5)
    for _ in range(5):
        current = residual_rmsnorm_cuda_vectorized(x, residual, weight, 1.0e-5)
        torch.testing.assert_close(current, first, rtol=0.0, atol=0.0)


def test_operator_respects_non_default_cuda_stream() -> None:
    torch.manual_seed(2029)
    x = torch.randn((2, 32, 768), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(768, device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = residual_rmsnorm_cuda_vectorized(x, residual, weight, 1.0e-5)
    stream.synchronize()
    expected = residual_rmsnorm_reference(x, residual, weight, 1.0e-5)
    torch.testing.assert_close(actual, expected, rtol=1.0e-3, atol=1.0e-3)


def test_non_contiguous_input_is_rejected() -> None:
    base = torch.randn((4, 8, 64), device="cuda", dtype=torch.float16)
    x = base.transpose(0, 1)
    residual = torch.randn_like(x)
    weight = torch.randn(64, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="contiguous"):
        residual_rmsnorm_cuda_vectorized(x, residual, weight, 1.0e-5)
