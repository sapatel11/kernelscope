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
    """Run the original shared-memory reduction baseline."""

    return _C.residual_rmsnorm_cuda_naive(input_tensor, residual, weight, epsilon)


def residual_rmsnorm_cuda_warp(
    input_tensor: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    """Run the scalar warp-shuffle optimized RMSNorm baseline."""

    return _C.residual_rmsnorm_cuda(input_tensor, residual, weight, epsilon)


def residual_rmsnorm_cuda_vectorized(
    input_tensor: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    """Run half2 RMSNorm when possible, with scalar warp fallback for odd widths."""

    return _C.residual_rmsnorm_cuda_vectorized(
        input_tensor, residual, weight, epsilon
    )


def residual_rmsnorm_cuda(
    input_tensor: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    """Run KernelScope's preferred RMSNorm implementation.

    The native vectorized operator uses packed half2 memory access for compatible
    even hidden dimensions and falls back to the proven scalar warp-reduction
    implementation otherwise.
    """

    return residual_rmsnorm_cuda_vectorized(
        input_tensor, residual, weight, epsilon
    )


def swiglu_cuda(gate: Tensor, value: Tensor) -> Tensor:
    """Run the fused CUDA SwiGLU implementation."""

    return _C.swiglu_cuda(gate, value)
