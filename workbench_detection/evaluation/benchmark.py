"""Inference timing and model size (proposal section 4.6)."""

import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Sequence

# %% benchmark
TIMING_SCOPE = ("Decoded image in memory through preprocessing, device transfers, forward pass, "
                "box decoding and NMS to final CPU boxes and scores")




def benchmark_inference(predict_one: Callable[[Any], Any], samples: Sequence[Any], *,
                        warmup: int = 5, repeats: int = 3,
                        synchronize: Callable[[], Any] | None = None,
                        device: str, timed_scope: str = TIMING_SCOPE,
                        precision: str | None = None) -> dict:
    """Batch-one wall-clock timing. Adapter must use eval/inference mode itself.

    Call with synchronize=torch.cuda.synchronize for CUDA.
    Supply preloaded samples unless image loading is intentionally part of scope.
    State whether preprocessing, device copies, postprocessing/NMS are included.
    The returned predictions are not evaluated or saved by this helper.
    Do not claim FPS here is batched/asynchronous throughput.
    """
    if (len(samples) == 0 or not isinstance(warmup, int) or isinstance(warmup, bool)
            or not isinstance(repeats, int) or isinstance(repeats, bool)
            or warmup < 0 or repeats < 1 or not device.strip() or not timed_scope.strip()):
        raise ValueError("Need samples, nonnegative warmup, positive repeats, device and timing scope.")
    if "cuda" in device.lower() and synchronize is None:
        raise ValueError("CUDA timing requires a synchronize callback.")
    sync = synchronize or (lambda: None)
    for index in range(warmup):
        predict_one(samples[index % len(samples)])
    sync()
    times_ms = []
    for _ in range(repeats):
        for sample in samples:
            sync()
            start = time.perf_counter()
            predict_one(sample)
            sync()
            times_ms.append((time.perf_counter() - start) * 1000)
    mean_ms = statistics.mean(times_ms)
    ordered = sorted(times_ms)
    return {"device": device, "precision": precision, "timed_scope": timed_scope, "batch_size": 1,
            "warmup_calls": warmup, "measured_calls": len(times_ms),
            "mean_latency_ms": mean_ms, "median_latency_ms": statistics.median(times_ms),
            "p95_latency_ms_nearest_rank": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "sequential_fps": 1000 / mean_ms if mean_ms else None,
            "latencies_ms": times_ms}


def model_size_summary(model: Any = None, checkpoint_path: str | Path | None = None) -> dict:
    """Optional duck-typed PyTorch model; no torch import is needed by this file."""
    summary = {}
    if model is not None:
        parameters = list(model.parameters())
        summary.update(parameter_count=sum(p.numel() for p in parameters),
                       trainable_parameter_count=sum(p.numel() for p in parameters if p.requires_grad),
                       parameter_storage_bytes=sum(p.numel() * p.element_size() for p in parameters))
    if checkpoint_path is not None:
        summary["checkpoint_file_bytes"] = Path(checkpoint_path).stat().st_size
    return summary
