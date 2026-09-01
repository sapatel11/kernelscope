"""Small transformer-style inference stack for end-to-end KernelScope demos."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .cuda import residual_rmsnorm_cuda, swiglu_cuda
from .reference import residual_rmsnorm_reference, swiglu_reference


class KernelScopeFeedForwardBlock(nn.Module):
    """Llama-style feed-forward block with eager and custom-kernel paths.

    The block keeps matrix multiplication in ``nn.Linear`` so CUDA execution is
    delegated to PyTorch/cuBLAS. Only residual RMSNorm and SwiGLU differ between
    the eager and optimized paths.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.epsilon = epsilon

        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.output_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def _project(self, normalized: Tensor, *, optimized: bool) -> Tensor:
        gate = self.gate_proj(normalized)
        value = self.value_proj(normalized)
        activated = (
            swiglu_cuda(gate, value)
            if optimized
            else swiglu_reference(gate, value)
        )
        return self.output_proj(activated)

    def forward_eager(self, input_tensor: Tensor, residual: Tensor) -> tuple[Tensor, Tensor]:
        """Run the PyTorch eager reference path."""

        normalized = residual_rmsnorm_reference(
            input_tensor,
            residual,
            self.norm_weight,
            self.epsilon,
        )
        combined_residual = input_tensor + residual
        branch_output = self._project(normalized, optimized=False)
        return branch_output, combined_residual

    def forward_optimized(
        self,
        input_tensor: Tensor,
        residual: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the custom CUDA RMSNorm and SwiGLU path."""

        normalized = residual_rmsnorm_cuda(
            input_tensor,
            residual,
            self.norm_weight,
            self.epsilon,
        )
        combined_residual = input_tensor + residual
        branch_output = self._project(normalized, optimized=True)
        return branch_output, combined_residual

    def forward(
        self,
        input_tensor: Tensor,
        residual: Tensor,
        *,
        optimized: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if optimized:
            return self.forward_optimized(input_tensor, residual)
        return self.forward_eager(input_tensor, residual)


class KernelScopeMiniTransformer(nn.Module):
    """Four-layer transformer-style feed-forward stack for inference benchmarks."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layers: int = 4,
        epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("layers must be positive")

        self.layers = nn.ModuleList(
            KernelScopeFeedForwardBlock(hidden_size, intermediate_size, epsilon)
            for _ in range(layers)
        )

    def _run(self, input_tensor: Tensor, *, optimized: bool) -> Tensor:
        branch = input_tensor
        residual = torch.zeros_like(input_tensor)

        for layer in self.layers:
            if optimized:
                branch, residual = layer.forward_optimized(branch, residual)
            else:
                branch, residual = layer.forward_eager(branch, residual)

        return branch + residual

    def forward_eager(self, input_tensor: Tensor) -> Tensor:
        return self._run(input_tensor, optimized=False)

    def forward_optimized(self, input_tensor: Tensor) -> Tensor:
        return self._run(input_tensor, optimized=True)

    def forward(self, input_tensor: Tensor, *, optimized: bool = False) -> Tensor:
        return self._run(input_tensor, optimized=optimized)
