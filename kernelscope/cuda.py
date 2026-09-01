"""Thin Python wrappers around KernelScope's compiled CUDA operators."""

from torch import Tensor

try:
    from . import _C
except ImportError as error:
    raise ImportError(
        "KernelScope's native extension is not built. Run "
        "`.venv\\Scripts\\python.exe setup.py build_ext --inplace` "
        "from the project root in a Visual Studio x64 developer terminal."
    ) from error


def residual_rmsnorm_cuda_naive(
    input_tensor: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    """Run the readable naive CUDA residual + RMSNorm baseline."""

    return _C.residual_rmsnorm_cuda_naive(
        input_tensor,
        residual,
        weight,
        epsilon,
    )


def residual_rmsnorm_cuda(
    input_tensor: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    """Run the warp-reduced CUDA residual + RMSNorm implementation."""

    return _C.residual_rmsnorm_cuda(
        input_tensor,
        residual,
        weight,
        epsilon,
    )


def swiglu_cuda(gate: Tensor, value: Tensor) -> Tensor:
    """Run the fused CUDA SwiGLU implementation."""

    return _C.swiglu_cuda(gate, value)
