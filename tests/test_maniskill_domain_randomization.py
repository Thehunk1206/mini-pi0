"""Tests for ManiSkill domain randomization config and scripted placement targets."""

from __future__ import annotations

import numpy as np
import pytest

from mini_pi0.dataset.maniskill_collectors.policy import ScriptedMultiObjectOracle
from mini_pi0.sim.domain_randomization import parse_domain_randomization_config


def test_parse_domain_randomization_missing_returns_disabled_config() -> None:
    # Act
    cfg = parse_domain_randomization_config(None)

    # Assert
    assert cfg.enabled is False
    assert cfg.camera.enabled is False
    assert cfg.physics.object_mass_scale_range == (1.0, 1.0)


def test_parse_domain_randomization_invalid_range_raises_value_error() -> None:
    # Arrange
    raw = {"enabled": True, "physics": {"object_friction_range": [1.3, 0.7]}}

    # Act / Assert
    with pytest.raises(ValueError, match="object_friction_range"):
        parse_domain_randomization_config(raw)


def test_scripted_oracle_uses_per_object_place_target() -> None:
    # Arrange
    policy = ScriptedMultiObjectOracle(tray_center=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    obs = {
        "robot0_eef_pos": np.array([0.25, 0.25, 0.20], dtype=np.float32),
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
        "observation.state.object": np.array([[0.05, -0.20, 0.03]], dtype=np.float32).reshape(-1),
        "observation.state.object_mask": np.array([1.0], dtype=np.float32),
        "observation.state.placed_mask": np.array([0.0], dtype=np.float32),
        "observation.state.grasped_mask": np.array([1.0], dtype=np.float32),
        "observation.state.place_targets": np.array([[0.04, 0.31, 0.065]], dtype=np.float32).reshape(-1),
    }
    policy.target_idx = 0
    policy.phase = "to_tray_above"
    policy.phase_step = 0

    # Act
    action = policy.act(obs)

    # Assert
    assert action[1] > 0.0
