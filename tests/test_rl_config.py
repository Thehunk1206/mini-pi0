"""Behavior tests for ReinFlow configuration validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mini_pi0.config.io import load_config
from mini_pi0.rl.config import validate_rl_config


_REINFLOW_CONFIGS = (
    "isaaclab_franka_lift_reinflow_finetune.yaml",
    "isaaclab_franka_lift_reinflow_scratch.yaml",
    "maniskill3_peginsertion_reinflow_finetune.yaml",
    "maniskill3_peginsertion_reinflow_potential.yaml",
    "maniskill3_pickcube_reinflow_finetune.yaml",
    "maniskill3_pickcube_reinflow_scratch.yaml",
)


def test_reinflow_scratch_does_not_require_artifacts() -> None:
    cfg = load_config(
        overrides=[
            "rl.algorithm='reinflow_ppo'",
            "rl.init_mode='scratch'",
            "rl.action_normalization='env_bounds'",
            "rl.checkpoint=None",
            "rl.action_stats_path=None",
        ]
    )

    validate_rl_config(cfg, require_files=True)


def test_reinflow_checkpoint_validates_matching_action_stats(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    stats = tmp_path / "action_stats.json"
    checkpoint.write_bytes(b"placeholder")
    stats.write_text(json.dumps({"mean": [0.0] * 7, "std": [1.0] * 7}), encoding="utf-8")
    cfg = load_config(
        overrides=[
            "rl.algorithm='reinflow_ppo'",
            "rl.init_mode='checkpoint'",
            "rl.action_normalization='dataset_stats'",
            f"rl.checkpoint='{checkpoint}'",
            f"rl.action_stats_path='{stats}'",
            "rl.use_reference_policy=True",
        ]
    )

    validate_rl_config(cfg)


@pytest.mark.parametrize("algorithm", ["flow_ppo", "ppo"])
def test_legacy_algorithm_names_are_rejected_with_migration_message(algorithm: str) -> None:
    cfg = load_config(overrides=[f"rl.algorithm='{algorithm}'"])

    with pytest.raises(ValueError, match="reinflow_ppo|ambiguous"):
        validate_rl_config(cfg, require_files=False)


def test_unknown_algorithm_is_rejected() -> None:
    cfg = load_config(overrides=["rl.algorithm='sac'"])

    with pytest.raises(ValueError, match="rl.algorithm"):
        validate_rl_config(cfg, require_files=False)


def test_gaussian_baseline_requires_checkpoint_mode() -> None:
    cfg = load_config(overrides=["rl.algorithm='gaussian_ppo_baseline'", "rl.init_mode='scratch'"])

    with pytest.raises(ValueError, match="checkpoint"):
        validate_rl_config(cfg, require_files=False)


def test_scratch_rejects_reference_regularization() -> None:
    cfg = load_config(
        overrides=[
            "rl.algorithm='reinflow_ppo'",
            "rl.init_mode='scratch'",
            "rl.reference_w2_coef=0.1",
            "rl.use_reference_policy=True",
        ]
    )

    with pytest.raises(ValueError, match="Scratch ReinFlow"):
        validate_rl_config(cfg, require_files=False)


@pytest.mark.parametrize(
    ("minimum", "initial", "maximum"),
    [(0.0, 0.05, 0.1), (0.08, 0.05, 0.1), (0.05, 0.2, 0.1)],
)
def test_invalid_noise_bounds_are_rejected(minimum: float, initial: float, maximum: float) -> None:
    cfg = load_config(
        overrides=[
            f"rl.noise_std_min={minimum}",
            f"rl.noise_std_init={initial}",
            f"rl.noise_std_max={maximum}",
        ]
    )

    with pytest.raises(ValueError, match="Noise bounds"):
        validate_rl_config(cfg, require_files=False)


def test_resume_is_exclusive_with_warm_start_checkpoint(tmp_path: Path) -> None:
    cfg = load_config(
        overrides=[
            f"rl.resume_from='{tmp_path / 'resume.pt'}'",
            f"rl.checkpoint='{tmp_path / 'bc.pt'}'",
        ]
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_rl_config(cfg, require_files=False)


@pytest.mark.parametrize("filename", _REINFLOW_CONFIGS)
def test_reinflow_example_config_is_valid(filename: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "examples" / "configs" / filename)

    validate_rl_config(cfg, require_files=False)
