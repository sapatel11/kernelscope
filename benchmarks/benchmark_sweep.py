from __future__ import annotations

import csv
import platform
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from kernelscope import KernelScopeMiniTransformer


WARMUP_ITERATIONS = 20
MEASURED_ITERATIONS = 50
LAYERS = 4
DTYPE = torch.float16
RESULTS_PATH = Path(__file__).with_name("results_transformer.csv")


@dataclass(frozen=True)
class Workload:
    batch: int
    sequence: int
    hidden: int
    intermediate: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.batch, self.sequence, self.hidden)

    @property
    def tokens(self) -> int:
        return self.batch * self.sequence


WORKLOADS = (
    Workload(1, 128, 512, 1408),
    Workload(1, 512, 768, 2048),
    Workload(8, 128, 768, 2048),
    Workload(16, 256, 1024, 2816),
)


def _measure_samples(operation, iterations: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for start, end in zip(starts, ends, strict=True):
        start.record()
        operation()
        end.record()

    ends[-1].synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _summarize(samples: list[float]) -> tuple[float, float, float]:
    return (
        statistics.fmean(samples),
        _percentile(samples, 0.50),
        _percentile(samples, 0.95),
    )


def _benchmark_workload(workload: Workload, device: torch.device) -> dict[str, float | int | str]:
    torch.manual_seed(2026)
    model = KernelScopeMiniTransformer(
        workload.hidden,
        workload.intermediate,
        layers=LAYERS,
    ).to(device=device, dtype=DTYPE).eval()
    input_tensor = torch.randn(workload.shape, device=device, dtype=DTYPE)

    with torch.inference_mode():
        eager = lambda: model.forward_eager(input_tensor)
        optimized = lambda: model.forward_optimized(input_tensor)

        eager()
        optimized()
        torch.cuda.synchronize()

        for _ in range(WARMUP_ITERATIONS):
            eager()
            optimized()
        torch.cuda.synchronize()

        eager_samples = _measure_samples(eager, MEASURED_ITERATIONS)
        optimized_samples = _measure_samples(optimized, MEASURED_ITERATIONS)

        eager_output = eager()
        optimized_output = optimized()
        torch.cuda.synchronize()

    eager_mean, eager_p50, eager_p95 = _summarize(eager_samples)
    optimized_mean, optimized_p50, optimized_p95 = _summarize(optimized_samples)
    max_abs_error = (optimized_output.float() - eager_output.float()).abs().max().item()

    return {
        "batch": workload.batch,
        "sequence": workload.sequence,
        "hidden": workload.hidden,
        "intermediate": workload.intermediate,
        "layers": LAYERS,
        "eager_mean_ms": eager_mean,
        "eager_p50_ms": eager_p50,
        "eager_p95_ms": eager_p95,
        "optimized_mean_ms": optimized_mean,
        "optimized_p50_ms": optimized_p50,
        "optimized_p95_ms": optimized_p95,
        "speedup": eager_mean / optimized_mean,
        "eager_tokens_per_second": workload.tokens / (eager_mean / 1000.0),
        "optimized_tokens_per_second": workload.tokens / (optimized_mean / 1000.0),
        "max_abs_error": max_abs_error,
    }


def _write_csv(rows: list[dict[str, float | int | str]]) -> None:
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)

    print("KernelScope Stage 6 — Multi-shape transformer benchmark")
    print("=" * 72)
    print(f"GPU: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"dtype: {DTYPE}")
    print(f"layers: {LAYERS}")
    print(f"warmup iterations: {WARMUP_ITERATIONS}")
    print(f"measured iterations per path: {MEASURED_ITERATIONS}")
    print()

    rows: list[dict[str, float | int | str]] = []
    for workload in WORKLOADS:
        result = _benchmark_workload(workload, device)
        rows.append(result)
        print(
            f"shape={workload.shape}, intermediate={workload.intermediate}: "
            f"eager={result['eager_mean_ms']:.4f} ms "
            f"(p50={result['eager_p50_ms']:.4f}, p95={result['eager_p95_ms']:.4f}), "
            f"optimized={result['optimized_mean_ms']:.4f} ms "
            f"(p50={result['optimized_p50_ms']:.4f}, p95={result['optimized_p95_ms']:.4f}), "
            f"speedup={result['speedup']:.3f}x, "
            f"throughput={result['optimized_tokens_per_second']:,.0f} tok/s, "
            f"max_err={result['max_abs_error']:.8g}"
        )

        del result
        torch.cuda.empty_cache()

    _write_csv(rows)
    print()
    print(f"CSV written to: {RESULTS_PATH}")
    print("All performance values are measurements from the machine that executed this script.")


if __name__ == "__main__":
    main()
