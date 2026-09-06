from __future__ import annotations

import threading
import time

import numpy as np
import torch
from lerobot.policies.rtc import RTCConfig

from so101.policy_inference.config import (
    InferenceConfig,
    RTCInferenceConfig,
    SafetyConfig,
)
from so101.policy_inference.inference_engine import AsyncInferenceEngine
from so101.policy_inference.policy_bundle import JOINT_NAMES
from so101.policy_inference.safety import PolicySafetyGate


class _FakeBundle:
    chunk_size = 8
    action_dim = 6
    device = torch.device("cpu")

    def preprocess(self, cameras, state):
        return torch.zeros(1), torch.as_tensor(state).unsqueeze(0)

    def denormalize_actions(self, actions):
        return actions

    def normalize_actions(self, actions):
        return actions


class _FakePolicy:
    def __init__(
        self, *, wait: threading.Event | None = None, rtc_enabled: bool = True
    ) -> None:
        self.rtc_config = RTCConfig(enabled=rtc_enabled)
        self.wait = wait
        self.started = threading.Event()
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def sample_normalized(self, images, state, *, previous_leftover, inference_delay):
        self.started.set()
        if self.wait is not None:
            self.wait.wait(timeout=2.0)
        return torch.full((1, 8, 6), 0.5)


def _engine(policy: _FakePolicy, *, rtc_enabled: bool = True) -> AsyncInferenceEngine:
    config = InferenceConfig(
        checkpoint=None,  # type: ignore[arg-type]
        control_hz=30.0,
        rtc=RTCInferenceConfig(enabled=rtc_enabled, flow_steps=1, replan_interval=2),
        safety=SafetyConfig(camera_stale_s=1.0),
    )
    limits = {
        name: ((0.0, 100.0) if name == "gripper" else (-100.0, 100.0))
        for name in JOINT_NAMES
    }
    safety = PolicySafetyGate(limits, control_hz=30.0, config=config.safety)
    measured = np.asarray([0, 0, 0, 0, 0, 50], dtype=np.float32)
    safety.reset(measured)
    return AsyncInferenceEngine(_FakeBundle(), config, safety, policy=policy)  # type: ignore[arg-type]


def _publish(engine: AsyncInferenceEngine) -> np.ndarray:
    measured = np.asarray([0, 0, 0, 0, 0, 50], dtype=np.float32)
    engine.publish_observation(
        {"wrist": np.zeros((4, 4, 3), np.uint8), "base": np.zeros((4, 4, 3), np.uint8)},
        measured,
    )
    return measured


def test_worker_fills_queue_and_control_consumes_safe_action() -> None:
    engine = _engine(_FakePolicy())
    engine.start()
    measured = _publish(engine)
    deadline = time.monotonic() + 2.0
    while engine.status.queue_size == 0 and time.monotonic() < deadline:
        time.sleep(0.005)

    action = engine.get_action(measured)
    engine.stop()

    assert action is not None
    assert action.shape == (6,)
    assert engine.status.inference_count == 1


def test_reset_drops_inflight_chunk() -> None:
    release = threading.Event()
    policy = _FakePolicy(wait=release)
    engine = _engine(policy)
    engine.start()
    measured = _publish(engine)
    assert policy.started.wait(timeout=1.0)

    engine.pause(measured, reason="operator_pause")
    release.set()
    deadline = time.monotonic() + 2.0
    while engine.status.stale_results_dropped == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    status = engine.status
    engine.stop()

    assert status.stale_results_dropped == 1
    assert status.queue_size == 0
    assert status.paused


def test_non_rtc_append_does_not_apply_latency_offset() -> None:
    engine = _engine(_FakePolicy(rtc_enabled=False), rtc_enabled=False)
    engine.start()
    measured = _publish(engine)
    deadline = time.monotonic() + 2.0
    while engine.status.inference_count < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert engine.get_action(measured) is not None
    assert engine.get_action(measured) is not None

    _publish(engine)
    deadline = time.monotonic() + 2.0
    while engine.status.inference_count < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    status = engine.status
    engine.stop()

    assert status.inference_count == 2
    assert status.last_predicted_delay == 0
    assert status.last_real_delay == 0
