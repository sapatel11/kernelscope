# KernelScope

KernelScope is a compact CUDA transformer-inference optimization project built around two custom operators:

- fused residual addition + RMSNorm
- fused SwiGLU

It demonstrates the full performance-engineering loop: establish a numerical reference, implement a readable CUDA baseline, profile it, optimize the bottleneck, test an additional vectorized memory path, integrate the operators with PyTorch/cuBLAS, and measure both kernel-level and end-to-end effects.

## What this project demonstrates

- CUDA C++ kernels and PyTorch C++/CUDA extensions
- FP16 inputs/outputs with FP32 accumulation
- block/thread mapping and reductions
- shared-memory tree reduction
- warp-shuffle reduction with `__shfl_down_sync`
- packed FP16 (`half2`) memory access with safe scalar fallback
- kernel fusion
- PyTorch current-stream integration and launch checking
- correctness tests across odd/even hidden sizes and non-default streams
- CUDA-event benchmarking with repeated-round medians
- Nsight Systems timeline analysis
- Nsight Compute hardware-counter analysis
- cuBLAS-backed `nn.Linear` integration in a small transformer-style stack
- lightweight CI/source-integrity checks, with CUDA tests explicitly requiring NVIDIA hardware

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
4. scalar optimized CUDA implementation using warp-shuffle reduction
5. vectorized CUDA implementation using `half2` loads/stores plus the same warp-reduction strategy
6. preferred dispatcher that selects the vectorized path for even hidden sizes and safely falls back to the scalar warp path for odd hidden sizes

CUDA storage is FP16 while residual arithmetic, reduction, normalization, and scaling use FP32 before the final cast back to FP16.

Public CUDA APIs:

```python
residual_rmsnorm_cuda_naive(...)       # readable baseline
residual_rmsnorm_cuda_warp(...)        # scalar warp-reduced baseline
residual_rmsnorm_cuda_vectorized(...)  # explicit half2 path + odd-width fallback
residual_rmsnorm_cuda(...)             # preferred dispatcher
```

### SwiGLU

```text
output = SiLU(gate) * value
```

The CUDA version fuses SiLU and multiplication into one elementwise kernel.

## Measured RMSNorm progression

All measurements below were produced on the RTX 4060 Laptop GPU above; they are not estimated or synthetic.

### Stage 3: shared-memory baseline to warp reduction

Representative shape `(8, 128, 768)`:

| Implementation | Mean latency |
| --- | ---: |
| PyTorch eager formula | 0.149750 ms |
| Naive CUDA | 0.016460 ms |
| Warp-reduced CUDA | 0.014781 ms |

- eager / warp: **10.131x**
- naive / warp: **1.114x**
- maximum absolute error vs PyTorch reference: **0.0078125**

The large eager-to-CUDA gap comes primarily from fusion and fewer kernel launches. The smaller naive-to-warp gain isolates the effect of the reduction optimization itself.

### Stage 9: vectorized `half2` experiment

To avoid drawing conclusions from one timing sample, the vectorization experiment used 30 warmup iterations, 100 iterations per round, five rounds, and the **median across rounds** as the comparison metric.

| Shape | Naive median | Warp median | `half2` median | Warp / `half2` | Naive / `half2` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(1, 128, 512)` | 0.014715 ms | 0.015063 ms | **0.012944 ms** | **1.164x** | **1.137x** |
| `(1, 512, 768)` | 0.019609 ms | 0.022354 ms | **0.014104 ms** | **1.585x** | **1.390x** |
| `(8, 128, 768)` | 0.029184 ms | 0.032154 ms | **0.023153 ms** | **1.389x** | **1.261x** |
| `(16, 256, 1024)` | 0.103782 ms | 0.069929 ms | **0.053021 ms** | **1.319x** | **1.957x** |

Maximum absolute error remained **0.0078125** on all four workloads.

The experiment therefore justified promoting `half2` as the preferred path for even hidden dimensions. The scalar warp implementation remains available as both an explicit comparison baseline and the fallback for odd widths.

## Fused SwiGLU result

Representative shape `(8, 128, 2048)`:

| Implementation | Mean latency |
| --- | ---: |
| PyTorch eager SwiGLU | 0.053842 ms |
| Fused CUDA SwiGLU | 0.027818 ms |

- eager / fused: **1.936x**
- maximum absolute error: **0.0078125**

## Mini-transformer integration

`KernelScopeMiniTransformer` is a four-layer transformer-style feed-forward stack. Both paths share the same FP16 `nn.Linear` layers, so GEMMs remain on the standard PyTorch/cuBLAS path. Only residual RMSNorm and SwiGLU differ.

With the promoted vectorized RMSNorm dispatcher, shape `(8, 128, 768)`, intermediate size `2048`, four layers:

| Path | Block latency | 4-layer latency | Throughput |
| --- | ---: | ---: | ---: |
| PyTorch eager | 0.717869 ms | 3.169137 ms | 323,116.4 tokens/s |
| KernelScope optimized | 0.736707 ms | **2.737080 ms** | **374,121.3 tokens/s** |

- single-block ratio: **0.974x** (slightly slower in this run)
- four-layer speedup: **1.158x**
- maximum absolute error: **0.005859375**

The single-block result is intentionally retained. A faster microkernel does not guarantee every higher-level timing improves on every run because GEMMs, launch scheduling, clocks, thermals, and other runtime effects dominate much of the block. The four-layer measurement still improves in the same run.

## Multi-shape benchmark with the promoted default

Four-layer FP16 inference using the vectorized dispatcher where valid:

| Shape | Intermediate | Eager mean | Optimized mean | Mean speedup | Optimized throughput | Max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(1, 128, 512)` | 1408 | 1.8488 ms | **0.6335 ms** | **2.918x** | 202,050 tok/s | 0.00390625 |
| `(1, 512, 768)` | 2048 | 2.0628 ms | **1.3498 ms** | **1.528x** | 379,302 tok/s | 0.005859375 |
| `(8, 128, 768)` | 2048 | 2.8156 ms | **2.3434 ms** | **1.202x** | 436,974 tok/s | 0.005859375 |
| `(16, 256, 1024)` | 2816 | 23.3583 ms | **14.9193 ms** | **1.566x** | 274,544 tok/s | 0.005859375 |

The optimized path was faster by mean latency on all four workloads in this final sweep. The benchmark also records p50 and p95 because earlier runs exposed substantial tail-latency variance; the project therefore avoids treating a single mean as a universal hardware claim.

## Nsight Systems evidence

A profiled four-layer run used NVTX ranges for eager and optimized paths before the later `half2` promotion.

- eager NVTX range: **31.066 ms**
- optimized NVTX range: **22.566 ms**
- profiled-region ratio: about **1.38x**

The CUDA kernel summary showed GEMMs dominating total GPU time. The eager range also launched separate kernels for add, SiLU, multiply, power/square, mean reduction, rsqrt, copies/conversions, and scaling. The optimized range replaced the relevant operation sequences with `residual_rmsnorm_fused_kernel` and `swiglu_fused_kernel` while leaving cuBLAS GEMMs unchanged.

This supports the architectural conclusion that KernelScope reduces launch and intermediate-operation overhead but cannot remove the dominant GEMM cost.

## Nsight Compute evidence: naive vs scalar warp RMSNorm

The Nsight Compute comparison isolates the Stage 3 reduction optimization; it predates the later `half2` memory-access experiment. Representative launch: grid `1024`, block size `256`, hidden size `768`.

| Metric | Naive | Scalar warp |
| --- | ---: | ---: |
| Kernel duration | 23.78 us | **20.48 us** |
| Instructions executed | 2,220,032 | **1,656,832** |
| Achieved occupancy | 91.77% | **96.19%** |
| Registers / thread | 18 | 18 |
| User reduction shared memory | ~1 KB dynamic | **32 B static** |
| Memory throughput | 132.61 GB/s | **154.29 GB/s** |

The naive implementation stores one FP32 partial sum per thread and repeatedly performs block-wide shared-memory reduction steps. The warp implementation reduces within each warp using shuffle instructions and stores only eight warp totals in shared memory.

The profiler shows the expected consequences: fewer executed instructions, much less reduction shared-memory traffic, slightly higher occupancy, and lower kernel duration. After this reduction optimization the kernel becomes more memory-heavy than compute-heavy, which motivated the later packed-memory experiment. The Stage 9 median results then showed that reducing memory-instruction overhead with `half2` provided additional measurable gains.

See `docs/optimization_report.md` for the full interpretation.

## Robustness and CI

The CUDA tests now cover representative tiny, odd, even, and large hidden dimensions; deterministic repeated execution; non-default CUDA streams; and validation failures such as non-contiguous input. The preferred RMSNorm dispatcher is tested to match the explicit vectorized path on even widths and the scalar warp path on odd widths.

The GitHub Actions workflow performs source/Python integrity checks on hosted runners. Native CUDA correctness and performance tests are deliberately not presented as CI-validated without actual NVIDIA hardware.

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
python benchmarks\benchmark_rmsnorm_vectorized.py
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
cuda/           naive, warp, vectorized RMSNorm and fused SwiGLU kernels
kernelscope/    Python API, references, dispatch, and transformer integration
benchmarks/     repeatable CUDA-event benchmarks
profiles/       Nsight profiling workloads
docs/           optimization report
tests/          CPU and CUDA correctness/validation tests
.github/        lightweight CI/source-integrity workflow
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

The custom CUDA API currently targets contiguous FP16 CUDA tensors and an Ada `sm_89` development machine. The `half2` path requires an even hidden dimension and otherwise falls back to the scalar warp implementation. Performance results are specific to the measured RTX 4060 Laptop GPU and should not be generalized to other hardware without remeasurement.

## Key takeaway

KernelScope demonstrates a complete GPU optimization workflow rather than only a fast kernel: establish a numerical contract, build a readable baseline, measure it, use profiler evidence to remove reduction/synchronization overhead, identify the resulting memory pressure, test packed memory access with a robust fallback, and then verify how much of those improvements survive integration into a GEMM-dominated transformer workload.
