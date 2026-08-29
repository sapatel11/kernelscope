"""Thin Python wrappers around the compiled C++ CPU reference extension."""

from torch import Tensor

try:
    from . import _C
except ImportError as error:
    raise ImportError(
        "KernelScope's CPU extension is not built. Run "
        "`.venv\\Scripts\\python.exe setup.py build_ext --inplace` "
        "from the project root in a Visual Studio developer terminal."
    ) from error


def residual_rmsnorm_cpu(
    input_tensor: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    """Run the manual C++ residual + RMSNorm CPU reference."""

    return _C.residual_rmsnorm_cpu(input_tensor, residual, weight, epsilon)


def swiglu_cpu(gate: Tensor, value: Tensor) -> Tensor:
    """Run the manual C++ SwiGLU CPU reference."""

    return _C.swiglu_cpu(gate, value)
