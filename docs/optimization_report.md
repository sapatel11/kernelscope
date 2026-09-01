# KernelScope Optimization Report

## Executive summary

KernelScope evaluates two transformer-inference operations—residual + RMSNorm and SwiGLU—through a correctness-first CUDA optimization workflow on an NVIDIA GeForce RTX 4060 Laptop GPU.

The project begins with CPU/PyTorch references, establishes a readable shared-memory CUDA RMSNorm baseline, replaces its block-wide reduction with warp-shuffle reduction, implements fused SwiGLU, integrates both custom operators into a four-layer transformer-style feed-forward stack, and validates the design with CUDA-event benchmarks plus Nsight Systems and Nsight Compute.

The optimized RMSNorm reduced representative kernel duration from 23.78 us to 20.48 us while cutting executed instructions from 2.22M to 1.66M and increasing achieved occupancy from 91.77% to 96.19%. Fused SwiGLU measured 1.936x faster than its PyTorch eager formula on the representative workload. In a four-layer `(8,128,768)` integration benchmark, the custom-operator path improved latency from 2.3095 ms to 2.1196 ms, a 1.090x end-to-end speedup.

The main systems conclusion is that fusion and reduction tuning substantially improve small transformer operators, but application-level gains are constrained by unchanged cuBLAS GEMMs.

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

This design is intentionally readable and leaves obvious optimization opportunities. It uses no warp shuffles, packed `half2` access, aggressive unrolling, shape-specific specialization, or custom streams.

## 4. Optimized RMSNorm design

The optimized kernel keeps the same block mapping, FP16/FP32 numerical contract, and second output pass. The change is concentrated in the reduction.

Each warp reduces its local FP32 partial sums with `__shfl_down_sync`. Only lane 0 of each warp writes a total to shared memory. With 256 threads, this means eight warp totals rather than 256 per-thread totals. The first warp then reduces those eight values with another shuffle reduction.

This changes the reduction structure from repeated block-wide shared-memory stages to:

- warp-local register exchange;
- eight shared-memory writes;
- one block synchronization;
- one warp reduction of eight totals;
- one final block synchronization before output.

The optimized implementation therefore targets synchronization and shared-memory overhead without changing the memory footprint of the input/output tensors or the mathematical result.

## 5. Kernel benchmark results

### Residual + RMSNorm

Representative shape: `(8,128,768)`, FP16.

| Implementation | Mean latency |
| --- | ---: |
| PyTorch eager formula | 0.149750 ms |
| Naive CUDA | 0.016460 ms |
| Optimized CUDA | 0.014781 ms |

The optimized kernel measured 10.131x faster than the PyTorch eager formula and 1.114x faster than the naive custom CUDA baseline.

The two speedups have different meanings. The large eager-to-custom difference mainly reflects operator fusion and launch reduction. The smaller naive-to-optimized difference isolates the effect of changing the reduction strategy.

### Fused SwiGLU

Representative shape: `(8,128,2048)`, FP16.

| Implementation | Mean latency |
| --- | ---: |
| PyTorch eager | 0.053842 ms |
| Fused CUDA | 0.027818 ms |

Measured eager/fused ratio: 1.936x.

This operator is elementwise, so its optimization story is simpler than RMSNorm: one custom kernel combines activation and multiplication instead of relying on separate eager operations.

## 6. End-to-end integration

KernelScope integrates the custom operators into a four-layer feed-forward transformer-style stack. Both paths use identical `nn.Linear` modules and therefore the same PyTorch/cuBLAS GEMM implementation. Only residual RMSNorm and SwiGLU differ.

For `(8,128,768)`, intermediate size 2048:

| Path | Block | Four layers | Throughput |
| --- | ---: | ---: | ---: |
| Eager | 0.442576 ms | 2.309549 ms | 443,376.6 tokens/s |
| Optimized | 0.395479 ms | 2.119557 ms | 483,119.8 tokens/s |

The block improved by 1.119x and the four-layer model by 1.090x.

This gap between kernel and application speedups is expected. Once the operators are embedded in the model, GEMMs consume most of the GPU time and remain unchanged.

## 7. Multi-shape behavior

Four-layer measurements:

| Shape | Eager mean | Optimized mean | Mean speedup |
| --- | ---: | ---: | ---: |
| `(1,128,512)` | 1.4572 ms | 0.6405 ms | 2.275x |
| `(1,512,768)` | 6.7140 ms | 2.6155 ms | 2.567x |
| `(8,128,768)` | 2.6753 ms | 3.3040 ms | 0.810x |
| `(16,256,1024)` | 21.4964 ms | 15.3214 ms | 1.403x |

The `(8,128,768)` sweep run is important because it demonstrates benchmark variance. Optimized median latency was faster than eager (`1.6220 ms` vs `2.3757 ms`), but optimized p95 reached `11.6582 ms`, raising the mean above eager.

This is a reason to keep p50/p95 data and profiler results rather than using only a single mean. The project does not claim that every run or shape is uniformly faster.

## 8. Nsight Systems findings

A profiled four-layer run used NVTX ranges around eager and optimized paths.

- eager range: 31.066 ms
- optimized range: 22.566 ms
- ratio: approximately 1.38x

The CUDA kernel summary showed that two FP16 GEMM kernel families accounted for roughly 87% of total kernel time. This confirms that matrix multiplication dominates the integrated workload.

The eager range included separate CUDA kernels for operations such as:

- residual addition
- power/square
- mean reduction
- rsqrt
- normalization/scaling
- SiLU
- multiply
- copies/conversions

The optimized range included KernelScope's `residual_rmsnorm_fused_kernel` and `swiglu_fused_kernel` around the same cuBLAS-backed GEMMs.

Therefore, Nsight Systems supports the interpretation that the custom operators reduce launch/intermediate-operation overhead while leaving the dominant GEMM work unchanged.

## 9. Nsight Compute findings

Representative RMSNorm launch configuration: grid 1024, block 256, hidden size 768.

| Metric | Naive | Optimized | Change |
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

The profiler evidence matches the implementation intent. The optimized kernel executes fewer instructions and avoids the baseline's dynamic shared-memory tree reduction. It maintains the same register count while achieving slightly better occupancy.

The optimized kernel also shifts toward a more memory-bound profile: Nsight Compute reports memory utilization exceeding compute utilization. This matters for future work because it suggests that additional reduction-only micro-optimization is unlikely to produce the same return; the remaining limit increasingly comes from reading/writing the tensor data itself.

## 10. Why the optimization works

The baseline's reduction repeatedly moves partial sums through shared memory and synchronizes the entire block. Warp shuffle instructions allow threads within a warp to exchange register values without using shared memory for each reduction stage.

The optimization therefore reduces:

- shared-memory reduction traffic;
- block-wide synchronization work;
- total executed instructions.

It does not reduce the fundamental input/residual/weight reads or output writes. That explains both the measurable improvement and the remaining memory bottleneck.

## 11. Alternatives not pursued

Several additional optimizations were intentionally excluded to keep the project focused:

- `half2`/packed memory access
- vectorized loads/stores
- shape-specific block-size tuning
- extensive template specialization
- CUDA Graphs
- Triton implementation
- TensorRT plugin
- custom GEMM

These could be useful follow-up experiments, but they are not required to demonstrate the core performance-engineering workflow.

## 12. Limitations

- Results are from one RTX 4060 Laptop GPU and should not be generalized without remeasurement.
- The custom API targets contiguous FP16 CUDA tensors.
- Only forward inference is implemented.
- The model is a small feed-forward transformer-style demonstration, not a production LLM runtime.
- GEMM and attention optimization are outside the project scope.
- Multi-shape timing showed substantial tail-latency variance on one workload.
- FP16 output rounding produces maximum absolute differences up to roughly 0.008 in measured operator tests; correctness is evaluated with combined absolute/relative tolerances against FP32-accumulating references.

## 13. Conclusion

KernelScope demonstrates the difference between three levels of performance work:

1. **fusion** can remove many eager launches and intermediate operations, producing large isolated speedups;
2. **kernel micro-optimization** such as warp-shuffle reduction produces smaller but measurable gains over a custom baseline;
3. **end-to-end integration** reveals how much of those gains remain once dominant operations such as GEMMs are included.

The project therefore provides a compact example of profile-guided CUDA engineering: correctness first, measurable baseline, targeted optimization, profiler validation, and honest application-level interpretation.
