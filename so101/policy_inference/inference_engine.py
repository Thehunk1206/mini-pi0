"""Asynchronous action-chunk inference and 30 Hz queue consumption."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import torch
from lerobot.policies.rtc import ActionQueue

from .config import InferenceConfig
from .policy_bundle import PolicyBundle
from .rtc_policy import MiniPi0RTCPolicy
from .safety import PolicySafetyGate, TrackingStatus, UnsafePolicyChunk


@dataclass(frozen=True)
class ObservationSnapshot:
    cameras: dict[str, np.ndarray]
    state: np.ndarray
    timestamp: float
    version: int


@dataclass(frozen=True)
class InferenceStatus:
    phase: str = "created"
    paused: bool = True
    fault: str | None = None
    queue_size: int = 0
    observation_version: int = -1
    inference_count: int = 0
    stale_results_dropped: int = 0
    rejected_chunks: int = 0
    rtc_fallbacks: int = 0
    underflow_count: int = 0
    last_latency_ms: float = 0.0
    last_predicted_delay: int = 0
    last_real_delay: int = 0
    last_inference_error: str | None = None
    last_rejection_reason: str | None = None


class AsyncInferenceEngine:
    """Run all accelerator work on one worker while control consumes a queue."""

    def __init__(
        self,
        bundle: PolicyBundle,
        config: InferenceConfig,
        safety: PolicySafetyGate,
        *,
        policy: MiniPi0RTCPolicy | None = None,
    ) -> None:
        self.bundle = bundle
        self.config = config
        self.safety = safety
        self.policy = policy or MiniPi0RTCPolicy(bundle, config.rtc)
        self.queue = ActionQueue(self.policy.rtc_config)
        self.refill_threshold = max(1, bundle.chunk_size - config.rtc.replan_interval)
        self._condition = threading.Condition()
        self._status = InferenceStatus()
        self._snapshot: ObservationSnapshot | None = None
        self._version = 0
        self._last_inferred_version = -1
        self._epoch = 0
        self._stopping = False
        self._worker: threading.Thread | None = None
        self._underflow_streak = 0
        self._has_ready_chunk = False
        self._force_unguided_once = False
        self._latest_raw_action: np.ndarray | None = None
        self._latest_tracking = TrackingStatus(
            warning=False,
            fault=False,
            worst_joint=None,
            worst_error=0.0,
            errors={name: 0.0 for name in self.safety.joint_limits},
        )

    @property
    def status(self) -> InferenceStatus:
        with self._condition:
            return replace(self._status, queue_size=self.queue.qsize())

    @property
    def latest_raw_action(self) -> np.ndarray | None:
        with self._condition:
            return (
                None
                if self._latest_raw_action is None
                else self._latest_raw_action.copy()
            )

    @property
    def latest_tracking(self) -> TrackingStatus:
        with self._condition:
            return self._latest_tracking

    def start(self) -> None:
        with self._condition:
            if self._worker is not None:
                raise RuntimeError("Inference engine is already started")
            self._stopping = False
            self._status = replace(
                self._status, phase="waiting_for_observation", paused=False, fault=None
            )
            self._worker = threading.Thread(
                target=self._worker_loop, name="so101-policy-inference", daemon=True
            )
            self._worker.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        with self._condition:
            self._stopping = True
            self._status = replace(self._status, phase="stopping", paused=True)
            self._condition.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout=timeout)
            if worker.is_alive():
                raise TimeoutError("Inference worker did not stop in time")
        with self._condition:
            self._worker = None
            self.queue.clear()
            self._status = replace(self._status, phase="stopped", queue_size=0)

    def publish_observation(
        self,
        cameras: Mapping[str, np.ndarray],
        state: np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> None:
        """Publish a copied latest-only observation for the inference worker."""

        camera_copy = {
            name: np.asarray(frame).copy() for name, frame in cameras.items()
        }
        state_copy = np.asarray(state, dtype=np.float32).reshape(-1).copy()
        with self._condition:
            self._version += 1
            self._snapshot = ObservationSnapshot(
                cameras=camera_copy,
                state=state_copy,
                timestamp=time.monotonic() if timestamp is None else float(timestamp),
                version=self._version,
            )
            self._status = replace(self._status, observation_version=self._version)
            self._condition.notify_all()

    def reset(
        self, measured: np.ndarray, *, reason: str = "reset", resume: bool = False
    ) -> None:
        """Invalidate in-flight inference and clear both action queues."""

        self.safety.reset(measured)
        with self._condition:
            self._epoch += 1
            self.queue.clear()
            self.policy.reset()
            self._last_inferred_version = -1
            self._underflow_streak = 0
            self._has_ready_chunk = False
            self._force_unguided_once = False
            self._latest_raw_action = None
            self._status = replace(
                self._status,
                phase="waiting_for_observation" if resume else "paused",
                paused=not resume,
                fault=None if resume else reason,
                queue_size=0,
                underflow_count=0,
            )
            self._condition.notify_all()

    def pause(self, measured: np.ndarray, *, reason: str) -> None:
        self.reset(measured, reason=reason, resume=False)

    def resume(self, measured: np.ndarray) -> None:
        self.reset(measured, reason="resume", resume=True)

    def get_action(
        self, measured: np.ndarray, *, now: float | None = None
    ) -> np.ndarray | None:
        """Consume one safe action; return ``None`` when the arm must hold."""

        measured_values = np.asarray(measured, dtype=np.float32).reshape(-1)
        current_time = time.monotonic() if now is None else float(now)
        with self._condition:
            if self._status.paused:
                return None
            snapshot = self._snapshot
        if (
            snapshot is None
            or current_time - snapshot.timestamp > self.config.safety.camera_stale_s
        ):
            self.pause(measured_values, reason="camera_stale")
            return None

        tracking = self.safety.evaluate_tracking(measured_values)
        with self._condition:
            self._latest_tracking = tracking
        if tracking.fault:
            self.pause(
                measured_values,
                reason=f"following_error:{tracking.worst_joint}:{tracking.worst_error:.2f}",
            )
            return None

        action = self.queue.get()
        if action is None:
            with self._condition:
                if self._has_ready_chunk:
                    self._underflow_streak += 1
                    self._status = replace(
                        self._status,
                        phase="queue_underflow",
                        underflow_count=self._status.underflow_count + 1,
                    )
                    should_fault = (
                        self._underflow_streak
                        >= self.config.safety.underflow_fault_cycles
                    )
                else:
                    self._status = replace(self._status, phase="warming_up")
                    should_fault = False
            if should_fault:
                self.pause(measured_values, reason="queue_underflow")
            return None

        values = action.detach().cpu().numpy().astype(np.float32, copy=False)
        self.safety.record_command(values)
        with self._condition:
            self._underflow_streak = 0
            self._status = replace(
                self._status, phase="running", queue_size=self.queue.qsize()
            )
            self._condition.notify_all()
        return values.copy()

    def tracking_status(self, measured: np.ndarray) -> TrackingStatus:
        del measured
        return self.latest_tracking

    def _ready_to_infer(self) -> bool:
        return (
            not self._stopping
            and not self._status.paused
            and self._snapshot is not None
            and self._snapshot.version != self._last_inferred_version
            and self.queue.qsize() <= self.refill_threshold
        )

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stopping or self._ready_to_infer(), timeout=0.1
                )
                if self._stopping:
                    return
                if not self._ready_to_infer():
                    continue
                assert self._snapshot is not None
                snapshot = self._snapshot
                epoch = self._epoch
                self._last_inferred_version = snapshot.version
                # Only RTC replaces the active queue and guides against its
                # unconsumed prefix. In ordinary chunking mode the next chunk
                # is appended intact, so applying an inference-delay offset
                # would incorrectly discard or duplicate its leading actions.
                queue_previous = (
                    self.queue.get_left_over() if self.config.rtc.enabled else None
                )
                using_unguided_fallback = (
                    self._force_unguided_once and queue_previous is not None
                )
                guidance_previous = None if using_unguided_fallback else queue_previous
                action_index = self.queue.get_action_index()
                predicted_delay = (
                    0
                    if queue_previous is None
                    else max(
                        1,
                        math.ceil(
                            self._status.last_latency_ms
                            * self.config.control_hz
                            / 1000.0
                        ),
                    )
                )
                self._status = replace(
                    self._status,
                    phase="inferring",
                    last_predicted_delay=predicted_delay,
                    last_inference_error=None,
                )

            started = time.perf_counter()
            try:
                images, state = self.bundle.preprocess(snapshot.cameras, snapshot.state)
                normalized = self.policy.sample_normalized(
                    images,
                    state,
                    previous_leftover=guidance_previous,
                    inference_delay=predicted_delay,
                )
                if self.bundle.device.type == "mps":
                    torch.mps.synchronize()
                elif self.bundle.device.type == "cuda":
                    torch.cuda.synchronize(self.bundle.device)
                normalized_cpu = normalized.squeeze(0).detach().float().cpu()
                denormalized = self.bundle.denormalize_actions(normalized_cpu).numpy()
                latency_ms = (time.perf_counter() - started) * 1000.0
            except UnsafePolicyChunk as exc:
                with self._condition:
                    self._status = replace(
                        self._status,
                        phase="chunk_rejected",
                        rejected_chunks=self._status.rejected_chunks + 1,
                        last_inference_error=str(exc),
                        last_rejection_reason=str(exc),
                    )
                    self._force_unguided_once = (
                        queue_previous is not None and self.config.rtc.enabled
                    )
                continue
            except Exception as exc:  # noqa: BLE001 - worker must convert failures into a safe pause
                with self._condition:
                    self._epoch += 1
                    self.queue.clear()
                    self._status = replace(
                        self._status,
                        phase="inference_fault",
                        paused=True,
                        fault=f"inference:{type(exc).__name__}",
                        last_inference_error=str(exc),
                    )
                continue

            with self._condition:
                if epoch != self._epoch or self._status.paused:
                    self._status = replace(
                        self._status,
                        stale_results_dropped=self._status.stale_results_dropped + 1,
                    )
                    continue
                consumed = max(0, self.queue.get_action_index() - action_index)
                latency_delay = math.ceil(latency_ms * self.config.control_hz / 1000.0)
                real_delay = (
                    0 if queue_previous is None else max(consumed, latency_delay)
                )
                if real_delay >= len(denormalized):
                    self._status = replace(
                        self._status,
                        phase="chunk_expired",
                        rejected_chunks=self._status.rejected_chunks + 1,
                        last_latency_ms=latency_ms,
                        last_real_delay=real_delay,
                        last_inference_error="Inference latency consumed the complete action chunk",
                    )
                    continue

            try:
                self.safety.validate_chunk(denormalized)
                safe_remaining = self.safety.process_chunk(
                    denormalized[real_delay:], snapshot.state
                )
                processed = np.empty_like(denormalized)
                if real_delay:
                    processed[:real_delay] = safe_remaining[0]
                processed[real_delay:] = safe_remaining
            except UnsafePolicyChunk as exc:
                with self._condition:
                    self._status = replace(
                        self._status,
                        phase="chunk_rejected",
                        rejected_chunks=self._status.rejected_chunks + 1,
                        last_inference_error=str(exc),
                        last_rejection_reason=str(exc),
                    )
                    self._force_unguided_once = (
                        queue_previous is not None and self.config.rtc.enabled
                    )
                continue

            try:
                with self._condition:
                    if epoch != self._epoch or self._status.paused:
                        self._status = replace(
                            self._status,
                            stale_results_dropped=self._status.stale_results_dropped
                            + 1,
                        )
                        continue
                    processed_tensor = torch.from_numpy(processed)
                    # RTC must anchor to the plan the robot will actually consume.
                    # Calibration saturation and slew limiting are nonlinear, so
                    # retaining the raw pre-safety chunk here can make successive
                    # guidance drift away from executed motion.
                    safe_normalized = self.bundle.normalize_actions(processed_tensor)
                    self.queue.merge(
                        original_actions=safe_normalized,
                        processed_actions=processed_tensor,
                        real_delay=real_delay,
                        # Wall-clock latency is authoritative, matching LeRobot's
                        # RTC engine. Its optional index check is only diagnostic
                        # and becomes noisy when a dry-run loop is not perfectly
                        # phase-aligned with the inference thread.
                        action_index_before_inference=None,
                    )
                    self._has_ready_chunk = True
                    self._force_unguided_once = False
                    self._latest_raw_action = denormalized[real_delay].copy()
                    self._status = replace(
                        self._status,
                        phase="ready",
                        queue_size=self.queue.qsize(),
                        inference_count=self._status.inference_count + 1,
                        rtc_fallbacks=self._status.rtc_fallbacks
                        + int(using_unguided_fallback),
                        last_latency_ms=latency_ms,
                        last_real_delay=real_delay,
                    )
                    self._condition.notify_all()
            except Exception as exc:  # noqa: BLE001 - queue failures must not kill the worker silently
                with self._condition:
                    self._epoch += 1
                    self.queue.clear()
                    self._status = replace(
                        self._status,
                        phase="inference_fault",
                        paused=True,
                        fault=f"queue_merge:{type(exc).__name__}",
                        last_inference_error=str(exc),
                    )
