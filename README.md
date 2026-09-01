# KernelScope

KernelScope is a correctness-first CUDA optimization study built around two
transformer inference operations:

- fused residual addition and RMSNorm;
- fused SwiGLU activation.

The project progresses from readable CPU references to naive CUDA kernels,
profile-guided optimization, PyTorch integration, and end-to-end transformer
benchmarks. GEMM remains delegated to PyTorch/cuBLAS.

## Current status

Stage 1 establishes single-threaded C++ CPU correctness references for residual
RMSNorm and SwiGLU.

Stage 2 adds a readable naive CUDA residual + RMSNorm baseline. It maps one CUDA
block to each flattened tensor row, uses 256 threads per block, reduces FP32
partial sums through shared memory, accepts FP16 input/residual/weight tensors,
and writes FP16 output. The kernel deliberately avoids warp-shuffle reduction,
`half2`, vectorized loads, shape-specific tuning, and other later optimizations.

Confirmed local environment:

- Windows, NVIDIA GeForce RTX 4060 Laptop GPU (compute capability 8.9)
- CUDA Toolkit 13.2
- Visual Studio 2022 / MSVC 19.44
- Python 3.14
- PyTorch 2.13.0+cu132
- C++20

## Terminal workflow

KernelScope uses its project-local virtual environment only from the terminal;
no VS Code interpreter selection is required.

From a normal `cmd.exe` terminal, initialize the Visual Studio x64 developer
environment first:

```bat
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
```

Then, from the repository root, build and test with the project-local Python:

```bat
.venv\Scripts\python.exe setup.py build_ext --inplace
.venv\Scripts\python.exe -m pytest -q
```

The build targets compute capability 8.9 (`sm_89`) by default and uses C++20.
On Windows, `/Zc:preprocessor` is applied to MSVC and forwarded through NVCC.

## Stage 1 operations

For a final dimension of size `H`, residual RMSNorm computes:

```text
combined = input + residual
mean_square = sum(combined[i]^2) / H
output[i] = combined[i] / sqrt(mean_square + epsilon) * weight[i]
```

SwiGLU computes elementwise:

```text
output = SiLU(gate) * value
```

The CPU implementation accepts contiguous CPU `float32` tensors. It accumulates
RMSNorm's sum of squares in `double`, while the trusted PyTorch formula uses FP32
accumulation to mirror the CUDA numerical contract. Across inspected hidden
sizes from 8 through 1024, the largest Stage 1 absolute difference was
approximately `9.54e-7`.

## Stage 2 naive CUDA residual + RMSNorm

Public API:

```python
from kernelscope import residual_rmsnorm_cuda_naive

output = residual_rmsnorm_cuda_naive(
    input_tensor,
    residual,
    weight,
    1.0e-5,
)
```

Stage 2 requires contiguous CUDA `float16` tensors. `input_tensor` and
`residual` must have identical shapes, `weight` must be one-dimensional and
match the final hidden dimension, all tensors must be on the same CUDA device,
and epsilon must be finite and positive.

The kernel performs residual arithmetic, the sum-of-squares reduction,
normalization, and scaling in FP32 before rounding the output to FP16. Tests
compare it against `residual_rmsnorm_reference`, which follows the same FP32
accumulation contract.

### Run the Stage 2 latency benchmark

Build the extension first, then run:

```bat
.venv\Scripts\python.exe benchmarks\benchmark_rmsnorm_naive.py
```

The benchmark uses CUDA events after warmup and prints:

- GPU name and compute capability
- platform and Python version
- PyTorch and CUDA runtime versions
- NVCC version
- exact tensor shape and dtype
- warmup and measured iteration counts
- PyTorch eager latency
- naive CUDA operator latency
- eager/naive timing ratio
- maximum absolute numerical error

The benchmark intentionally does not contain hard-coded performance claims.
Published latency or speedup numbers must come from an actual run on the target
GPU.

## Scope

KernelScope is an educational inference-performance project, not a complete LLM
runtime. It does not implement custom GEMM, training, autograd, distributed
inference, or backward kernels. TensorRT and Triton are optional future work and
are not required for the core project.
