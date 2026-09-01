"""KernelScope CPU references, CUDA operators, and transformer demo."""

from .cpu import residual_rmsnorm_cpu, swiglu_cpu
from .cuda import (
    residual_rmsnorm_cuda,
    residual_rmsnorm_cuda_naive,
    residual_rmsnorm_cuda_vectorized,
    swiglu_cuda,
)
from .reference import residual_rmsnorm_reference, swiglu_reference
from .transformer import KernelScopeFeedForwardBlock, KernelScopeMiniTransformer

__all__ = [
    "KernelScopeFeedForwardBlock",
    "KernelScopeMiniTransformer",
    "residual_rmsnorm_cpu",
    "residual_rmsnorm_cuda",
    "residual_rmsnorm_cuda_naive",
    "residual_rmsnorm_cuda_vectorized",
    "residual_rmsnorm_reference",
    "swiglu_cpu",
    "swiglu_cuda",
    "swiglu_reference",
]
