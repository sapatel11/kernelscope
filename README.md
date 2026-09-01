# KernelScope

KernelScope is a compact CUDA transformer-inference optimization project built around two custom operators:

- fused residual addition + RMSNorm
- fused SwiGLU

It demonstrates the full performance-engineering loop: build a correctness reference, implement a readable CUDA baseline, profile it, optimize the bottleneck, integrate with PyTorch/cuBLAS, and measure both kernel-level and end-to-end effects.

## What this project demonstrates

- CUDA C++ kernels and PyTorch C++/CUDA extensions
- FP16 inputs/outputs with FP32 accumulation
- block/thread mapping and reductions
- shared-memory tree reduction
- warp-shuffle reduction with `__shfl_down_sync`
- kernel fusion
- PyTorch current-stream integration and launch checking
- CUDA-event benchmarking
- Nsight Systems timeline analysis
- Nsight Compute hardware-counter analysis
- cuBLAS-backed `nn.Linear` integration in a small transformer-style stack

Custom GEMM is intentionally out of scope; matrix multiplication remains delegated to PyTorch/cuBLAS.

## Hardware and software used for measured results

- NVIDIA GeForce RTX 4060 Laptop GPU
- compute capability 8.9 (`sm_89`)
- Windows 11
- CUDA Toolkit 13.2
- NVCC 13.2
- Visual Studio 2022 / MSVC 19.44
- Python 3.14
- PyTorch 2.13.0+cu132
- C++20
- Nsight Systems 2025.6.3
- Nsight Compute 2026.1

## Implemented operators

### Residual + RMSNorm

For hidden dimension `H`:

```text
combined = input + residual
mean_square = sum(combined[i]^2) / H
output[i] = combined[i] / sqrt(mean_square + epsilon) * weight[i]
```

The project includes:

1. single-threaded C++ CPU reference
2. trusted PyTorch reference
3. naive CUDA baseline using a 256-thread shared-memory tree reduction
4. optimized CUDA implementation using warp-shuffle reduction and only warp-level partial sums in shared memory

CUDA storage is FP16 while residual arithmetic, reduction, normalization, and scaling use FP32 before the final cast back to FP16.

### SwiGLU

```text
output = SiLU(gate) * value
```

The CUDA version fuses SiLU and multiplication into one elementwise kernel.

## Measured kernel results

All numbers below were measured on the RTX 4060 Laptop GPU above; they are not estimated or synthetic.

### Residual + RMSNorm — shape `(8, 128, 768)`

| Implementation | Mean latency |
| --- | ---: |
| PyTorch eager formula | 0.149750 ms |
| Naive CUDA | 0.016460 ms |
| Optimized CUDA | 0.014781 ms |

- eager / optimized: **10.131x**
- naive / optimized: **1.114x**
- maximum absolute error vs PyTorch reference: **0.0078125**

The large eager-to-CUDA gap comes primarily from fusion and fewer kernel launches. The smaller 1.114x naive-to-optimized gain isolates the effect of the reduction optimization itself.

### Fused SwiGLU — shape `(8, 128, 2048)`

| Implementation | Mean latency |
| --- | ---: |
| PyTorch eager SwiGLU | 0.053842 ms |
| Fused CUDA SwiGLU | 0.027818 ms |

- eager / fused: **1.936x**
- maximum absolute error: **0.0078125**

## Mini-transformer integration

`KernelScopeMiniTransformer` is a four-layer transformer-style feed-forward stack. Both eager and optimized paths share the same FP16 `nn.Linear` layers, so GEMMs remain on the standard PyTorch/cuBLAS path. Only residual RMSNorm and SwiGLU differ.

For shape `(8, 128, 768)`, intermediate size `2048`, four layers:

| Path | Block latency | 4-layer latency | Throughput |
| --- | ---: | ---: | ---: |
| PyTorch eager | 0.442576 ms | 2.309549 ms | 443,376.6 tokens/s |
| KernelScope optimized | 0.395479 ms | 2.119557 ms | 483,119.8 tokens/s |

- block speedup: **1.119x**
- four-layer speedup: **1.090x**
- maximum absolute error: **0.005859375**

This is the central end-to-end lesson of the project: large isolated kernel improvements translate into smaller application-level improvements once GEMMs dominate execution time.

## Multi-shape benchmark

Four-layer FP16 inference was measured across representative workloads:

| Shape | Intermediate | Eager mean | Optimized mean | Mean speedup | Optimized throughput | Max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(1, 128, 512)` | 1408 | 1.4572 ms | 0.6405 ms | **2.275x** | 199,857 tok/s | 0.00390625 |
| `(1, 512, 768)` | 2048 | 6.7140 ms | 2.6155 ms | **2.567x** | 195,756 tok/s | 0.005859375 |
| `(8, 128, 768)` | 2048 | 2.6753 ms | 3.3040 ms | **0.810x** | 309,925 tok/s | 0.005859375 |
| `(16, 256, 1024)` | 2816 | 21.4964 ms | 15.3214 ms | **1.403x** | 267,338 tok/s | 0.005859375 |

The `(8, 128, 768)` sweep result is intentionally retained rather than hidden. Its optimized median was faster (`1.6220 ms` vs `2.3757 ms`), but optimized p95 rose to `11.6582 ms`, making the mean slower. This demonstrates why GPU performance claims should include distributions and profiler evidence rather than only a single mean.

## Nsight Systems evidence

A profiled four-layer run used NVTX ranges for eager and optimized paths.

- eager NVTX range: **31.066 ms**
- optimized NVTX range: **22.566 ms**
- profiled-region ratio: about **1.38x**

The CUDA kernel summary showed GEMMs dominating total GPU time. The eager range also launched separate kernels for add, SiLU, multiply, power/square, mean reduction, rsqrt, copies/conversions, and scaling. The optimized range replaced the relevant operation sequences with `residual_rmsnorm_fused_kernel` and `swiglu_fused_kernel` while leaving cuBLAS GEMMs unchanged.

This supports the architectural conclusion that KernelScope reduces launch and intermediate-operation overhead but cannot remove the dominant GEMM cost.

## Nsight Compute evidence: naive vs optimized RMSNorm

Representative launch: grid `1024`, block size `256`, hidden size `768`.

| Metric | Naive | Optimized |
| --- | ---: | ---: |
| Kernel duration | 23.78 us | **20.48 us** |
| Instructions executed | 2,220,032 | **1,656,832** |
| Achieved occupancy | 91.77% | **96.19%** |
| Registers / thread | 18 | 18 |
| User reduction shared memory | ~1 KB dynamic | **32 B static** |
| Memory throughput | 132.61 GB/s | **154.29 GB/s** |

The naive implementation stores one FP32 partial sum per thread and repeatedly performs block-wide shared-memory reduction steps. The optimized implementation reduces within each warp using shuffle instructions and stores only eight warp totals in shared memory.

The profiler shows the expected consequences: fewer executed instructions, much less reduction shared-memory traffic, slightly higher occupancy, and lower kernel duration. After this optimization the kernel is more memory-heavy than compute-heavy, so further reduction-only tuning has diminishing returns.

See `docs/optimization_report.md` for the full interpretation.

## Build and test

On Windows, build from a Visual Studio x64 developer environment:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && .venv\Scripts\python.exe setup.py build_ext --inplace'
```

Then run:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

The extension targets compute capability 8.9 by default through `TORCH_CUDA_ARCH_LIST=8.9`.

## Benchmarks

```powershell
python benchmarks\benchmark_rmsnorm_naive.py
python benchmarks\benchmark_rmsnorm_optimized.py
python benchmarks\benchmark_swiglu.py
python benchmarks\benchmark_transformer.py
python benchmarks\benchmark_sweep.py
```

The sweep writes `benchmarks/results_transformer.csv`.

## Profiling workloads

```powershell
python profiles\profile_systems.py
python profiles\profile_rmsnorm_compute.py
```

These scripts are intended to be launched under Nsight Systems and Nsight Compute respectively.

## Repository structure

```text
cpp/            C++ CPU references and pybind bindings
cuda/           naive/optimized RMSNorm and fused SwiGLU kernels
kernelscope/    Python API, references, and transformer integration
benchmarks/     repeatable CUDA-event benchmarks
profiles/       Nsight profiling workloads
reports/        generated profiler reports when run locally
docs/           optimization report
tests/          CPU and CUDA correctness/validation tests
```

## Scope and limitations

KernelScope is an educational inference-performance project, not a full LLM runtime. It does not implement:

- custom GEMM
- attention kernels
- training or backward kernels
- autograd integration
- distributed inference
- CUDA Graphs
- TensorRT plugins
- Triton kernels
- broad dtype/device portability

The custom CUDA API currently targets contiguous FP16 CUDA tensors and an Ada `sm_89` development machine. Performance results are specific to the measured RTX 4060 Laptop GPU and should not be generalized to other hardware without remeasurement.

## Key takeaway

KernelScope demonstrates a complete GPU optimization workflow rather than only a fast kernel: establish a numerical contract, build a readable baseline, measure it, use profiler evidence to identify avoidable reduction/synchronization work, optimize the kernel, and then verify how much of that improvement survives integration into a GEMM-dominated transformer workload.
