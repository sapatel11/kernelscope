import math

import pytest
import torch

from kernelscope import residual_rmsnorm_cuda_naive, residual_rmsnorm_reference


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the native CUDA operator tests",
)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 8),
        (3, 5, 16),
        (1, 4, 64),
        (1, 128, 512),
        (1, 512, 768),
        (2, 32, 1024),
    ],
)
def test_naive_cuda_rmsnorm_matches_pytorch(shape: tuple[int, ...]) -> None:
    torch.manual_seed(2026)
    device = torch.device("cuda")
    input_tensor = torch.randn(shape, device=device, dtype=torch.float16)
    residual = torch.randn(shape, device=device, dtype=torch.float16)
    weight = torch.randn(shape[-1], device=device, dtype=torch.float16)
    epsilon = 1.0e-5

    expected = residual_rmsnorm_reference(
        input_tensor,
        residual,
        weight,
        epsilon,
    )
    actual = residual_rmsnorm_cuda_naive(
        input_tensor,
        residual,
        weight,
        epsilon,
    )

    # The operator stores FP16 tensors but performs the residual arithmetic,
    # reduction, normalization, and weight scaling in FP32. A 1e-3 tolerance is
    # therefore tied to the final FP16 rounding scale rather than a loose CUDA
    # implementation tolerance.
    torch.testing.assert_close(actual, expected, rtol=1.0e-3, atol=1.0e-3)


def test_naive_cuda_rmsnorm_rejects_cpu_input() -> None:
    input_tensor = torch.ones((2, 8), dtype=torch.float16)
    residual = torch.zeros_like(input_tensor)
    weight = torch.ones(8, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="input must be a CUDA tensor"):
        residual_rmsnorm_cuda_naive(input_tensor, residual, weight, 1.0e-5)


def test_naive_cuda_rmsnorm_rejects_float32() -> None:
    device = torch.device("cuda")
    input_tensor = torch.ones((2, 8), device=device, dtype=torch.float32)
    residual = torch.zeros_like(input_tensor)
    weight = torch.ones(8, device=device, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="float16"):
        residual_rmsnorm_cuda_naive(input_tensor, residual, weight, 1.0e-5)


def test_naive_cuda_rmsnorm_rejects_noncontiguous_input() -> None:
    device = torch.device("cuda")
    input_tensor = torch.randn((8, 8), device=device, dtype=torch.float16).t()
    residual = torch.randn((8, 8), device=device, dtype=torch.float16)
    weight = torch.ones(8, device=device, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="input must be contiguous"):
        residual_rmsnorm_cuda_naive(input_tensor, residual, weight, 1.0e-5)


def test_naive_cuda_rmsnorm_rejects_shape_mismatch() -> None:
    device = torch.device("cuda")
    input_tensor = torch.ones((2, 8), device=device, dtype=torch.float16)
    residual = torch.ones((1, 8), device=device, dtype=torch.float16)
    weight = torch.ones(8, device=device, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="identical shapes"):
        residual_rmsnorm_cuda_naive(input_tensor, residual, weight, 1.0e-5)


def test_naive_cuda_rmsnorm_rejects_bad_weight_length() -> None:
    device = torch.device("cuda")
    input_tensor = torch.ones((2, 8), device=device, dtype=torch.float16)
    residual = torch.zeros_like(input_tensor)
    weight = torch.ones(7, device=device, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="weight length"):
        residual_rmsnorm_cuda_naive(input_tensor, residual, weight, 1.0e-5)


@pytest.mark.parametrize("epsilon", [0.0, -1.0e-5, math.inf, math.nan])
def test_naive_cuda_rmsnorm_rejects_invalid_epsilon(epsilon: float) -> None:
    device = torch.device("cuda")
    input_tensor = torch.ones((2, 8), device=device, dtype=torch.float16)
    residual = torch.zeros_like(input_tensor)
    weight = torch.ones(8, device=device, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="epsilon"):
        residual_rmsnorm_cuda_naive(input_tensor, residual, weight, epsilon)
