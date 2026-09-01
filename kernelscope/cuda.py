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
    return _C.residual_rmsnorm_cuda_naive(input_tensor, residual, weight, epsilon)


def residual_rmsnorm_cuda(
    input_tensor: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    return _C.residual_rmsnorm_cuda(input_tensor, residual, weight, epsilon)


def residual_rmsnorm_cuda_vectorized(
    input_tensor: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    """Run the half2 experiment; odd hidden sizes fall back to scalar optimized CUDA."""

    return _C.residual_rmsnorm_cuda_vectorized(
        input_tensor, residual, weight, epsilon
    )


def swiglu_cuda(gate: Tensor, value: Tensor) -> Tensor:
    return _C.swiglu_cuda(gate, value)
