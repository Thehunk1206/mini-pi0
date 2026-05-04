"""Tests for simplified ManiSkill oracle mixture collection helpers."""

from __future__ import annotations

from mini_pi0.config.schema import RootConfig
from mini_pi0.dataset.maniskill_oracle_mixture import _episode_quality_ok, _profile_specs, allocate_profile_counts


def test_allocate_profile_counts_uses_expected_500_demo_mix() -> None:
    # Arrange
    mix = {"core": 0.65, "recovery": 0.25, "suboptimal": 0.10}

    # Act
    counts = allocate_profile_counts(500, mix)

    # Assert
    assert counts == {"core": 325, "recovery": 125, "suboptimal": 50}


def test_episode_quality_ok_rejects_failed_episode() -> None:
    # Arrange
    final_info = {"success": False, "success_fraction": 0.5}

    # Act
    ok, reason = _episode_quality_ok(final_info, 100, [], reject_long=True, max_retries=4)

    # Assert
    assert ok is False
    assert reason == "not_success"


def test_episode_quality_ok_rejects_overlong_episode() -> None:
    # Arrange
    final_info = {"success": True, "success_fraction": 1.0, "oracle_retry_count": 0, "oracle_phase_timeout_count": 0}

    # Act
    ok, reason = _episode_quality_ok(final_info, 401, [180, 200], reject_long=True, max_retries=4)

    # Assert
    assert ok is False
    assert reason == "too_long"


def test_episode_quality_ok_accepts_successful_short_episode() -> None:
    # Arrange
    final_info = {"success": True, "success_fraction": 1.0, "oracle_retry_count": 1, "oracle_phase_timeout_count": 0}

    # Act
    ok, reason = _episode_quality_ok(final_info, 220, [210, 230], reject_long=True, max_retries=4)

    # Assert
    assert ok is True
    assert reason == "accepted"


def test_profile_specs_include_visible_bowl_escape_recovery() -> None:
    # Act
    specs = _profile_specs(3, {"core": 0.0, "recovery": 1.0, "suboptimal": 0.0}, "balanced")

    # Assert
    recovery = [spec for spec in specs if spec.profile.value == "recovery"][0]
    assert "bowl_escape" in recovery.perturbation_types


def test_dataset_collection_config_defaults_to_single_env() -> None:
    # Act
    cfg = RootConfig()

    # Assert
    assert cfg.dataset_collection.num_envs == 1
