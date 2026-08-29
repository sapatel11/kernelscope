# KernelScope

KernelScope is a correctness-first CUDA optimization study built around two
transformer inference operations:

- fused residual addition and RMSNorm;
- fused SwiGLU activation.

The project progresses from readable CPU references to naive CUDA kernels,
profile-guided optimization, PyTorch integration, and end-to-end transformer
benchmarks. GEMM remains delegated to PyTorch/cuBLAS.

## Current status

Stage 1 implements single-threaded C++ CPU references and compares them with
trusted PyTorch formulas. These references prioritize clarity and numerical
correctness rather than CPU performance.

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

From a Visual Studio x64 developer command prompt:

```bat
cd /d E:\Shail_PC\Shail\Study\Instant_Projects\kernelscope
.venv\Scripts\activate.bat
python setup.py build_ext --inplace
python -m pytest -q
deactivate
```

To turn a normal `cmd.exe` terminal into the required developer prompt first:

```bat
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
```

Activation is optional. The equivalent commands using the environment directly
are:

```bat
.venv\Scripts\python.exe setup.py build_ext --inplace
.venv\Scripts\python.exe -m pytest -q
```

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

The C++ implementation accepts contiguous CPU `float32` tensors during this
stage. FP16 inputs with FP32 accumulation are introduced with the CUDA kernels.

The CPU implementation accumulates RMSNorm's sum of squares in `double`, while
the PyTorch formula uses FP32 accumulation to mirror the future CUDA contract.
Across inspected hidden sizes from 8 through 1024, the largest observed absolute
error was `9.54e-7` for both operations. Tests use tight FP32 tolerances while
leaving room for valid changes in reduction order across compilers.

## Scope

KernelScope is an educational inference-performance project, not a complete LLM
runtime. It does not implement custom GEMM, training, autograd, distributed
inference, or backward kernels.
