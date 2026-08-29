import pytest
import torch

from kernelscope import (
    residual_rmsnorm_cpu,
    residual_rmsnorm_reference,
    swiglu_cpu,
    swiglu_reference,
)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 8),
        (3, 5, 16),
        (1, 4, 64),
    ],
)
def test_residual_rmsnorm_matches_pytorch(shape: tuple[int, ...]) -> None:
    generator = torch.Generator().manual_seed(2026)
    input_tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
    residual = torch.randn(shape, generator=generator, dtype=torch.float32)
    weight = torch.randn(shape[-1], generator=generator, dtype=torch.float32)
    epsilon = 1.0e-5

    expected = residual_rmsnorm_reference(
        input_tensor,
        residual,
        weight,
        epsilon,
    )
    actual = residual_rmsnorm_cpu(
        input_tensor,
        residual,
        weight,
        epsilon,
    )

    # The C++ oracle reduces in double while PyTorch reduces in float32. The
    # measured max absolute error through hidden size 1024 is below 1e-6.
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=2.0e-6)


@pytest.mark.parametrize(
    "shape",
    [
        (17,),
        (2, 7, 32),
        (4, 3, 5, 16),
    ],
)
def test_swiglu_matches_pytorch(shape: tuple[int, ...]) -> None:
    generator = torch.Generator().manual_seed(2026)
    gate = torch.randn(shape, generator=generator, dtype=torch.float32)
    value = torch.randn(shape, generator=generator, dtype=torch.float32)

    expected = swiglu_reference(gate, value)
    actual = swiglu_cpu(gate, value)

    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-7)


def test_residual_rmsnorm_rejects_incompatible_weight() -> None:
    input_tensor = torch.ones((2, 8), dtype=torch.float32)
    residual = torch.zeros_like(input_tensor)
    weight = torch.ones(7, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="weight length"):
        residual_rmsnorm_cpu(input_tensor, residual, weight, 1.0e-5)


def test_residual_rmsnorm_rejects_nonpositive_epsilon() -> None:
    input_tensor = torch.ones((2, 8), dtype=torch.float32)
    residual = torch.zeros_like(input_tensor)
    weight = torch.ones(8, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="epsilon"):
        residual_rmsnorm_cpu(input_tensor, residual, weight, 0.0)


def test_swiglu_rejects_noncontiguous_input() -> None:
    gate = torch.randn((4, 4), dtype=torch.float32).transpose(0, 1)
    value = torch.randn((4, 4), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="gate must be contiguous"):
        swiglu_cpu(gate, value)


def test_cpu_references_reject_float64() -> None:
    gate = torch.ones(8, dtype=torch.float64)
    value = torch.ones_like(gate)

    with pytest.raises(RuntimeError, match="float32"):
        swiglu_cpu(gate, value)
