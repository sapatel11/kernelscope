import pytest
import torch

from kernelscope import (
    residual_rmsnorm_cuda,
    residual_rmsnorm_cuda_vectorized,
    residual_rmsnorm_cuda_warp,
    residual_rmsnorm_reference,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for RMSNorm dispatch tests",
)


def _inputs(hidden_size: int):
    torch.manual_seed(2026 + hidden_size)
    shape = (2, 7, hidden_size)
    device = torch.device("cuda")
    input_tensor = torch.randn(shape, device=device, dtype=torch.float16)
    residual = torch.randn_like(input_tensor)
    weight = torch.randn(hidden_size, device=device, dtype=torch.float16)
    return input_tensor, residual, weight


def test_default_matches_vectorized_for_even_hidden_size() -> None:
    input_tensor, residual, weight = _inputs(768)
    epsilon = 1.0e-5

    preferred = residual_rmsnorm_cuda(input_tensor, residual, weight, epsilon)
    vectorized = residual_rmsnorm_cuda_vectorized(
        input_tensor, residual, weight, epsilon
    )
    expected = residual_rmsnorm_reference(input_tensor, residual, weight, epsilon)

    torch.testing.assert_close(preferred, vectorized, rtol=0.0, atol=0.0)
    torch.testing.assert_close(preferred, expected, rtol=1.0e-3, atol=1.0e-3)


def test_default_uses_safe_fallback_for_odd_hidden_size() -> None:
    input_tensor, residual, weight = _inputs(769)
    epsilon = 1.0e-5

    preferred = residual_rmsnorm_cuda(input_tensor, residual, weight, epsilon)
    vectorized_dispatch = residual_rmsnorm_cuda_vectorized(
        input_tensor, residual, weight, epsilon
    )
    warp = residual_rmsnorm_cuda_warp(input_tensor, residual, weight, epsilon)

    torch.testing.assert_close(preferred, vectorized_dispatch, rtol=0.0, atol=0.0)
    torch.testing.assert_close(preferred, warp, rtol=0.0, atol=0.0)
