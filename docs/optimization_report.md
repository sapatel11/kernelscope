# KernelScope Optimization Report

## Executive summary

KernelScope evaluates two transformer-inference operations—residual + RMSNorm and SwiGLU—through a correctness-first CUDA optimization workflow on an NVIDIA GeForce RTX 4060 Laptop GPU.

The project begins with CPU/PyTorch references, establishes a readable shared-memory CUDA RMSNorm baseline, replaces its block-wide reduction with warp-shuffle reduction, profiles that change with Nsight Compute, then adds a packed `half2` memory-access path after profiling shows the warp-reduced kernel becoming more memory-heavy. Fused SwiGLU is implemented separately, and both custom operators are integrated into a four-layer transformer-style feed-forward stack with cuBLAS-backed `nn.Linear` layers.

The scalar warp optimization reduced representative RMSNorm kernel duration from 23.78 us to 20.48 us while cutting executed instructions from 2.22M to 1.66M and increasing achieved occupancy from 91.77% to 96.19%. A later repeated-round experiment showed the `half2` path outperforming the scalar warp path on all four representative RMSNorm workloads by median latency. The final four-layer `(8,128,768)` benchmark measured 3.1691 ms eager versus 2.7371 ms optimized, a 1.158x end-to-end speedup in that run.

The main systems conclusion is that fusion and reduction tuning substantially improve small transformer operators, but application-level gains are constrained by unchanged cuBLAS GEMMs and normal GPU timing variance.

## 1. Problem

Transformer inference contains many non-GEMM operations surrounding matrix multiplications. Individually these operations can be small, but separate launches and intermediate tensor traffic create overhead.

KernelScope focuses on two representative cases:

- residual addition followed by RMSNorm
- SwiGLU activation expressed as `SiLU(gate) * value`

The goal is not to replace highly optimized GEMM libraries. Instead, the project asks whether small elementwise/reduction sequences can be fused and simplified enough to produce measurable kernel-level gains, and whether those gains survive integration into a transformer-style workload.

## 2. Numerical contract

Residual RMSNorm uses:

```text
combined = input + residual
mean_square = mean(combined^2, dim=-1)
output = combined / sqrt(mean_square + epsilon) * weight
```

CUDA input, residual, weight, and output storage are FP16. Residual arithmetic, sum-of-squares accumulation, normalization, and weight scaling are performed in FP32, after which results are rounded to FP16.

SwiGLU computes:

```text
output = SiLU(gate) * value
```

with FP16 storage and FP32 intermediate arithmetic in the custom CUDA operator.

Correctness is checked against trusted PyTorch references using tolerances tied to final FP16 rounding rather than arbitrary loose CUDA tolerances.

## 3. Baseline RMSNorm design

The naive CUDA baseline maps one flattened RMSNorm row to one block of 256 threads.

Each thread:

1. strides over columns in the row;
2. loads FP16 input/residual values;
3. converts to FP32 and accumulates a local sum of squares;
4. writes one FP32 partial sum to shared memory.

The block then performs a standard tree reduction in shared memory, using `__syncthreads()` between reduction stages. After thread 0 computes inverse RMS, threads make a second pass through the row to produce FP16 output.

This design is intentionally readable and leaves obvious optimization opportunities.

## 4. Scalar warp-reduced RMSNorm

The first optimization keeps the same block mapping, FP16/FP32 numerical contract, and second output pass. The change is concentrated in the reduction.

Each warp reduces its local FP32 partial sums with `__shfl_down_sync`. Only lane 0 of each warp writes a total to shared memory. With 256 threads, this means eight warp totals rather than 256 per-thread totals. The first warp then reduces those eight values with another shuffle reduction.

This changes the reduction structure from repeated block-wide shared-memory stages to:

- warp-local register exchange;
- eight shared-memory writes;
- one block synchronization;
- one warp reduction of eight totals;
- one final block synchronization before output.

The optimization therefore targets synchronization and shared-memory overhead without changing the mathematical result.

## 5. Nsight Compute evidence for the reduction optimization

Representative RMSNorm launch configuration: grid 1024, block 256, hidden size 768.

| Metric | Naive | Scalar warp | Change |
| --- | ---: | ---: | ---: |
| Duration | 23.78 us | 20.48 us | lower |
| Elapsed cycles | 44,787 | 38,538 | lower |
| Instructions executed | 2,220,032 | 1,656,832 | -25.4% |
| Achieved occupancy | 91.77% | 96.19% | higher |
| Active warps/SM | 44.05 | 46.17 | higher |
| Registers/thread | 18 | 18 | unchanged |
| Dynamic shared memory/block | 1.02 KB | 0 | removed |
| Static shared memory/block | 0 | 32 B | small warp totals |
| Memory throughput | 132.61 GB/s | 154.29 GB/s | higher |
| DRAM throughput utilization | 51.94% | 60.55% | higher |
| Compute throughput utilization | 68.70% | 47.65% | lower |

The profiler evidence matches the implementation intent. The optimized kernel executes fewer instructions and avoids the baseline's dynamic shared-memory tree reduction while maintaining the same register count.

The scalar warp kernel also shifts toward a more memory-heavy profile: Nsight Compute reports memory utilization exceeding compute utilization. That observation motivates the next experiment—packed memory access—because further reduction-only tuning is increasingly unlikely to address the dominant cost.

## 6. Vectorized `half2` experiment

The next RMSNorm experiment keeps the warp-reduction strategy but loads and stores FP16 values in packed pairs using `half2` when the hidden dimension is even. Odd hidden dimensions safely fall back to the scalar warp path.

The benchmark intentionally uses repeated rounds rather than one timing sample:

- warmup iterations: 30
- iterations per round: 100
- rounds: 5
- comparison metric: median across the five round averages

Results:

| Shape | Naive median | Warp median | `half2` median | Warp / `half2` | Naive / `half2` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(1,128,512)` | 0.014715 ms | 0.015063 ms | **0.012944 ms** | **1.164x** | **1.137x** |
| `(1,512,768)` | 0.019609 ms | 0.022354 ms | **0.014104 ms** | **1.585x** | **1.390x** |
| `(8,128,768)` | 0.029184 ms | 0.032154 ms | **0.023153 ms** | **1.389x** | **1.261x** |
| `(16,256,1024)` | 0.103782 ms | 0.069929 ms | **0.053021 ms** | **1.319x** | **1.957x** |

Maximum absolute error was `0.0078125` on all four workloads.

Because the vectorized path wins by the specified median metric on all four representative shapes, KernelScope promotes it as the preferred implementation for even hidden dimensions. The scalar warp implementation remains available explicitly and is used automatically as the fallback for odd hidden sizes.

This progression is important: the project did not assume vectorization would help. The change was introduced only after profiling indicated a stronger memory component, and it was promoted only after repeated measurement supported the decision.

## 7. Fused SwiGLU

Representative shape: `(8,128,2048)`, FP16.

| Implementation | Mean latency |
| --- | ---: |
| PyTorch eager | 0.053842 ms |
| Fused CUDA | 0.027818 ms |

Measured eager/fused ratio: **1.936x**.

This operator is elementwise, so its optimization story is simpler than RMSNorm: one custom kernel combines activation and multiplication instead of relying on separate eager operations.

## 8. End-to-end integration with the promoted dispatcher

KernelScope integrates the custom operators into a four-layer feed-forward transformer-style stack. Both paths use identical `nn.Linear` modules and therefore the same PyTorch/cuBLAS GEMM implementation. Only residual RMSNorm and SwiGLU differ.

For `(8,128,768)`, intermediate size 2048, using the promoted vectorized dispatcher:

| Path | Block | Four layers | Throughput |
| --- | ---: | ---: | ---: |
| Eager | 0.717869 ms | 3.169137 ms | 323,116.4 tokens/s |
| Optimized | 0.736707 ms | **2.737080 ms** | **374,121.3 tokens/s** |

- block ratio: **0.974x** (slightly slower in this run)
- four-layer speedup: **1.158x**
- maximum absolute error: **0.005859375**

The single-block regression is retained rather than hidden. A faster microkernel does not guarantee every higher-level timing improves on every run because GEMMs, scheduling, clocks, thermals, and runtime noise dominate much of the block execution. The four-layer path still improves in the same run.

## 9. Final multi-shape behavior

Four-layer measurements using the preferred dispatcher:

| Shape | Eager mean | Optimized mean | Mean speedup |
| --- | ---: | ---: | ---: |
| `(1,128,512)` | 1.8488 ms | 0.6335 ms | **2.918x** |
| `(1,512,768)` | 2.0628 ms | 1.3498 ms | **1.528x** |
| `(8,128,768)` | 2.8156 ms | 2.3434 ms | **1.202x** |
| `(16,256,1024)` | 23.3583 ms | 14.9193 ms | **1.566x** |

The optimized path is faster by mean latency on all four workloads in this final sweep. The benchmark also records p50 and p95 because earlier runs showed substantial tail-latency variance; conclusions therefore rely on distributions and repeatability rather than a universal single-number claim.

## 10. Nsight Systems findings

A profiled four-layer run used NVTX ranges around eager and optimized paths before the later `half2` promotion.

- eager range: 31.066 ms
- optimized range: 22.566 ms
- ratio: approximately 1.38x

The CUDA kernel summary showed that two FP16 GEMM kernel families accounted for roughly 87% of total kernel time. This confirms that matrix multiplication dominates the integrated workload.

The eager range included separate CUDA kernels for operations such as residual addition, power/square, mean reduction, rsqrt, normalization/scaling, SiLU, multiply, and copies/conversions. The optimized range included KernelScope's fused RMSNorm and SwiGLU kernels around the same cuBLAS-backed GEMMs.

Therefore, Nsight Systems supports the interpretation that the custom operators reduce launch/intermediate-operation overhead while leaving the dominant GEMM work unchanged.

## 11. Robustness and engineering safeguards

The final implementation adds coverage beyond representative happy-path shapes:

- tiny, odd, even, and large hidden dimensions
- deterministic repeated execution
- non-default CUDA streams
- non-contiguous input rejection
- dispatcher checks proving even widths select the vectorized result and odd widths match the scalar fallback

The GitHub Actions workflow performs source/Python integrity checks on hosted runners. Native CUDA correctness and performance are explicitly treated as requiring real NVIDIA hardware rather than being presented as CI-validated on CPU-only infrastructure.

## 12. Why the optimization sequence works

The project demonstrates three different performance levers:

1. **Fusion** removes eager launches and intermediate tensors.
2. **Warp reduction** removes shared-memory reduction traffic and synchronization work.
3. **Packed `half2` access** reduces memory-instruction overhead once profiling indicates the optimized reduction is increasingly memory constrained.

No single step eliminates the fundamental input/residual/weight reads or output writes, and none changes the dominant cuBLAS GEMMs in the integrated model. That explains both the measurable improvements and the diminishing application-level gains.

## 13. Alternatives intentionally not pursued

KernelScope still avoids turning into a full runtime. It does not add:

- custom GEMM
- attention kernels
- shape-specialized kernel families
- aggressive template/autotuning systems
- CUDA Graphs
- Triton implementations
- TensorRT plugins
- training/backward kernels
- distributed inference

These are reasonable future projects, but they are not required to demonstrate the profile-guided CUDA engineering loop.

## 14. Limitations

- Results are from one RTX 4060 Laptop GPU and should not be generalized without remeasurement.
- The custom API targets contiguous FP16 CUDA tensors.
- The preferred `half2` path requires even hidden dimensions and otherwise falls back to the scalar warp kernel.
- Only forward inference is implemented.
- The model is a small feed-forward transformer-style demonstration, not a production LLM runtime.
- GEMM and attention optimization are outside the project scope.
- Timing distributions show that GPU runtime variance can materially affect individual benchmark runs.
- FP16 output rounding produces maximum absolute differences up to roughly 0.008 in measured operator tests; correctness is evaluated with combined absolute/relative tolerances against FP32-accumulating references.

## 15. Conclusion

KernelScope demonstrates a complete profile-guided GPU optimization workflow:

1. establish a trusted numerical contract;
2. build a readable shared-memory CUDA baseline;
3. measure and profile it;
4. replace block-wide reduction work with warp shuffles;
5. use profiler evidence to identify the next memory-related bottleneck;
6. test packed `half2` access with a correctness-preserving fallback;
7. promote the new path only after repeated measurements justify it;
8. integrate the custom operators into a GEMM-dominated transformer workload and report the smaller, more realistic application-level gains.

The result is not just a collection of fast kernels. It is a compact demonstration of how CUDA performance engineering decisions can be driven by correctness, measurement, profiling evidence, and honest end-to-end interpretation.
