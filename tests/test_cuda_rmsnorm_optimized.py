import pytest
import torch

from kernelscope import residual_rmsnorm_cuda, residual_rmsnorm_cuda_naive, residual_rmsnorm_reference


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
        (8, 128, 768),
        (2, 32, 1024),
    ],
)
def test_optimized_cuda_rmsnorm_matches_pytorch_and_naive(shape: tuple[int, ...]) -> None:
    torch.manual_seed(2026)
    device = torch.device("cuda")
    input_tensor = torch.randn(shape, device=device, dtype=torch.float16)
    residual = torch.randn(shape, device=device, dtype=torch.float16)
    weight = torch.randn(shape[-1], device=device, dtype=torch.float16)
    epsilon = 1.0e-5

    expected = residual_rmsnorm_reference(input_tensor, residual, weight, epsilon)
    naive = residual_rmsnorm_cuda_naive(input_tensor, residual, weight, epsilon)
    optimized = residual_rmsnorm_cuda(input_tensor, residual, weight, epsilon)

    torch.testing.assert_close(optimized, expected, rtol=1.0e-3, atol=1.0e-3)
    torch.testing.assert_close(optimized, naive, rtol=1.0e-3, atol=1.0e-3)
