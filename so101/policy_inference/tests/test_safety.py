from __future__ import annotations

import numpy as np
import pytest

from so101.policy_inference.config import SafetyConfig
from so101.policy_inference.policy_bundle import JOINT_NAMES
from so101.policy_inference.safety import PolicySafetyGate, UnsafePolicyChunk


def _limits() -> dict[str, tuple[float, float]]:
    return {
        name: ((0.0, 100.0) if name == "gripper" else (-100.0, 100.0))
        for name in JOINT_NAMES
    }


def test_chunk_is_rejected_atomically_when_one_target_exceeds_calibration() -> None:
    gate = PolicySafetyGate(_limits(), control_hz=30.0, config=SafetyConfig())
    measured = np.zeros(6, dtype=np.float32)
    measured[-1] = 50.0
    gate.reset(measured)
    chunk = np.repeat(measured[None], 8, axis=0)
    chunk[6, 2] = 102.5

    with pytest.raises(UnsafePolicyChunk, match="elbow_flex"):
        gate.process_chunk(chunk, measured)


def test_small_boundary_overshoot_saturates_without_expanding_limits() -> None:
    gate = PolicySafetyGate(_limits(), control_hz=30.0, config=SafetyConfig())
    measured = np.asarray([0, 0, 0, 0, 0, 50], dtype=np.float32)
    gate.reset(measured)
    chunk = np.repeat(measured[None], 120, axis=0)
    chunk[:, 3] = 101.5

    safe = gate.process_chunk(chunk, measured)

    assert float(safe[:, 3].max()) == 100.0


def test_chunk_slew_is_bounded_from_last_command() -> None:
    config = SafetyConfig(
        arm_velocity_deg_s=(30.0, 30.0, 30.0, 30.0, 30.0),
        gripper_velocity_percent_s=60.0,
    )
    gate = PolicySafetyGate(_limits(), control_hz=30.0, config=config)
    measured = np.asarray([0, 0, 0, 0, 0, 50], dtype=np.float32)
    gate.reset(measured)
    target = np.asarray([20, -20, 20, -20, 20, 80], dtype=np.float32)

    safe = gate.process_chunk(np.repeat(target[None], 3, axis=0), measured)

    assert np.allclose(safe[0], [1, -1, 1, -1, 1, 52])
    assert np.allclose(safe[2], [3, -3, 3, -3, 3, 56])


def test_following_fault_requires_configured_consecutive_cycles() -> None:
    gate = PolicySafetyGate(
        _limits(),
        control_hz=30.0,
        config=SafetyConfig(following_fault_cycles=3),
    )
    measured = np.asarray([0, 0, 0, 0, 0, 50], dtype=np.float32)
    gate.reset(measured)
    gate.record_command(np.asarray([20, 0, 0, 0, 0, 50], dtype=np.float32))

    assert gate.evaluate_tracking(measured).warning
    assert not gate.evaluate_tracking(measured).fault
    assert gate.evaluate_tracking(measured).fault


def test_normal_tracking_resets_fault_streak() -> None:
    gate = PolicySafetyGate(
        _limits(),
        control_hz=30.0,
        config=SafetyConfig(following_fault_cycles=2),
    )
    measured = np.asarray([0, 0, 0, 0, 0, 50], dtype=np.float32)
    gate.reset(measured)
    gate.record_command(np.asarray([20, 0, 0, 0, 0, 50], dtype=np.float32))
    assert not gate.evaluate_tracking(measured).fault
    assert not gate.evaluate_tracking(np.asarray([20, 0, 0, 0, 0, 50])).fault
    assert not gate.evaluate_tracking(measured).fault
