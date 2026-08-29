"""Trusted PyTorch formulas used to validate native KernelScope operators."""

import torch
from torch import Tensor
from torch.nn import functional as functional


def residual_rmsnorm_reference(
    input_tensor: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    """Compute residual addition and RMSNorm with FP32 accumulation."""

    combined = input_tensor + residual
    accumulated = combined.float()
    mean_square = accumulated.square().mean(dim=-1, keepdim=True)
    normalized = accumulated * torch.rsqrt(mean_square + epsilon)
    return (normalized * weight.float()).to(dtype=input_tensor.dtype)


def swiglu_reference(gate: Tensor, value: Tensor) -> Tensor:
    """Compute the unfused PyTorch SwiGLU formula."""

    return functional.silu(gate) * value
