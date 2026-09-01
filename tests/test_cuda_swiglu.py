import pytest
import torch

from kernelscope import swiglu_cuda, swiglu_reference


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the native CUDA operator tests",
)


@pytest.mark.parametrize(
    "shape",
    [
        (17,),
        (2, 7, 32),
        (4, 3, 5, 16),
        (8, 128, 2048),
    ],
)
def test_swiglu_cuda_matches_pytorch(shape: tuple[int, ...]) -> None:
    torch.manual_seed(2026)
    device = torch.device("cuda")
    gate = torch.randn(shape, device=device, dtype=torch.float16)
    value = torch.randn(shape, device=device, dtype=torch.float16)

    expected = swiglu_reference(gate, value)
    actual = swiglu_cuda(gate, value)

    torch.testing.assert_close(actual, expected, rtol=1.0e-3, atol=1.0e-3)


def test_swiglu_cuda_rejects_cpu_input() -> None:
    gate = torch.ones(8, dtype=torch.float16)
    value = torch.ones_like(gate)

    with pytest.raises(RuntimeError, match="gate must be a CUDA tensor"):
        swiglu_cuda(gate, value)


def test_swiglu_cuda_rejects_float32() -> None:
    device = torch.device("cuda")
    gate = torch.ones(8, device=device, dtype=torch.float32)
    value = torch.ones_like(gate)

    with pytest.raises(RuntimeError, match="float16"):
        swiglu_cuda(gate, value)


def test_swiglu_cuda_rejects_noncontiguous_input() -> None:
    device = torch.device("cuda")
    gate = torch.randn((8, 8), device=device, dtype=torch.float16).t()
    value = torch.randn((8, 8), device=device, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="gate must be contiguous"):
        swiglu_cuda(gate, value)


def test_swiglu_cuda_rejects_shape_mismatch() -> None:
    device = torch.device("cuda")
    gate = torch.ones((2, 8), device=device, dtype=torch.float16)
    value = torch.ones((1, 8), device=device, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="identical shapes"):
        swiglu_cuda(gate, value)
