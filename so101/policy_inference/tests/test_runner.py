from __future__ import annotations

from pathlib import Path

import pytest

from so101.policy_inference import runner
from so101.policy_inference.config import InferenceConfig
from so101.teleop.control_stack import DEFAULT_JOINT_LIMITS_DEG


def test_offline_safety_uses_generic_limits_without_calibration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "MOTOR_CALIBRATION_PATH", tmp_path / "missing.json")

    limits, _safety = runner._load_safety(
        InferenceConfig(checkpoint=tmp_path / "unused.pt")
    )

    assert limits == DEFAULT_JOINT_LIMITS_DEG


def test_hardware_safety_requires_calibration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(runner, "MOTOR_CALIBRATION_PATH", missing)

    with pytest.raises(FileNotFoundError, match=str(missing)):
        runner._load_safety(
            InferenceConfig(checkpoint=tmp_path / "unused.pt"),
            require_calibration=True,
        )
