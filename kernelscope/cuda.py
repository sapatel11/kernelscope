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
