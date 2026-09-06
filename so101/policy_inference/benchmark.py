"""Hardware-free latency benchmark for SO-101 policy variants."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import RTCInferenceConfig
from .policy_bundle import PolicyBundle, checkpoint_for_variant
from .rtc_policy import MiniPi0RTCPolicy


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    return {
        "mean_ms": float(statistics.fmean(ordered)),
        "p50_ms": float(np.percentile(ordered, 50)),
        "p95_ms": float(np.percentile(ordered, 95)),
        "p99_ms": float(np.percentile(ordered, 99)),
        "max_ms": float(max(ordered)),
    }


def benchmark_policy(
    *,
    checkpoint: Path,
    device: str,
    precision: str,
    flow_steps: int,
    repeats: int = 10,
    guided_repeats: int = 3,
    control_hz: float = 30.0,
) -> dict[str, Any]:
    """Benchmark preprocessing, ordinary sampling, and official RTC guidance."""

    if repeats <= 0 or guided_repeats <= 0:
        raise ValueError("benchmark repeat counts must be positive")
    bundle = PolicyBundle.load(checkpoint, device=device, precision=precision)
    cameras = {
        "wrist": np.zeros((480, 480, 3), dtype=np.uint8),
        "base": np.zeros((360, 640, 3), dtype=np.uint8),
    }
    state_array = bundle.normalization.state_mean.copy()
    bundle.warmup(flow_steps=min(2, flow_steps))

    preprocess_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        images, state = bundle.preprocess(cameras, state_array)
        _synchronize(bundle.device)
        preprocess_ms.append((time.perf_counter() - started) * 1000.0)

    noise = torch.zeros(
        1,
        bundle.chunk_size,
        bundle.action_dim,
        device=bundle.device,
    )
    ordinary_ms: list[float] = []
    ordinary = None
    for _ in range(repeats):
        started = time.perf_counter()
        ordinary = bundle.sample(
            images,
            state,
            flow_steps=flow_steps,
            solver="euler",
            initial_noise=noise,
        )
        _synchronize(bundle.device)
        ordinary_ms.append((time.perf_counter() - started) * 1000.0)
    assert ordinary is not None

    rtc_config = RTCInferenceConfig(
        enabled=True,
        flow_steps=flow_steps,
        execution_horizon=10,
        replan_interval=4,
        max_guidance_weight=5.0,
    )
    rtc = MiniPi0RTCPolicy(bundle, rtc_config)
    previous = ordinary[0, 4:].detach().cpu()
    guided_ms: list[float] = []
    for _ in range(guided_repeats):
        started = time.perf_counter()
        rtc.sample_normalized(
            images,
            state,
            previous_leftover=previous,
            inference_delay=1,
            initial_noise=noise,
        )
        _synchronize(bundle.device)
        guided_ms.append((time.perf_counter() - started) * 1000.0)

    guided_summary = _summary(guided_ms)
    control_period_ms = 1000.0 / control_hz
    estimated_delay_frames = int(np.ceil(guided_summary["p95_ms"] / control_period_ms))
    return {
        "checkpoint": str(bundle.checkpoint_path),
        "parameters": bundle.parameter_count,
        "device": str(bundle.device),
        "precision": bundle.precision_name,
        "flow_steps": flow_steps,
        "chunk_size": bundle.chunk_size,
        "control_hz": control_hz,
        "preprocess": _summary(preprocess_ms),
        "ordinary": _summary(ordinary_ms),
        "rtc_guided": guided_summary,
        "rtc_p95_delay_frames": estimated_delay_frames,
        "rtc_queue_feasible": estimated_delay_frames < bundle.chunk_size,
        "minimum_safe_replan_interval": estimated_delay_frames + 2,
    }


def benchmark_variants(
    *,
    variants: tuple[str, ...],
    device: str,
    precision: str,
    flow_steps: int,
    repeats: int,
    guided_repeats: int,
) -> list[dict[str, Any]]:
    return [
        benchmark_policy(
            checkpoint=checkpoint_for_variant(variant),
            device=device,
            precision=precision,
            flow_steps=flow_steps,
            repeats=repeats,
            guided_repeats=guided_repeats,
        )
        for variant in variants
    ]


def print_benchmark(results: list[dict[str, Any]]) -> None:
    print(json.dumps(results, indent=2))
