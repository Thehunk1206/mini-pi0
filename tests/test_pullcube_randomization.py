"""Tests for PullCube domain-randomization configuration."""

from __future__ import annotations

import pytest

from mini_pi0.sim.pullcube_randomization import parse_pullcube_domain_randomization


def test_pullcube_randomization_parses_nested_ranges() -> None:
    """Nested YAML sections should map to the flat typed runtime contract."""

    config = parse_pullcube_domain_randomization(
        {
            "enabled": True,
            "robot_init_qpos_noise": 0.05,
            "pose": {"cube_yaw_range_deg": [-45.0, 45.0]},
            "physics": {
                "cube_mass_scale_range": [0.7, 1.3],
                "friction_range": [0.5, 1.5],
            },
            "visual": {"color_jitter": 0.15},
        }
    )

    assert config.enabled
    assert config.robot_init_qpos_noise == pytest.approx(0.05)
    assert config.cube_yaw_range_deg == (-45.0, 45.0)
    assert config.cube_mass_scale_range == (0.7, 1.3)
    assert config.friction_range == (0.5, 1.5)
    assert config.color_jitter == pytest.approx(0.15)


@pytest.mark.parametrize(
    "raw",
    [
        {"physics": {"cube_mass_scale_range": [0.0, 1.0]}},
        {"physics": {"friction_range": [-0.1, 1.0]}},
        {"physics": {"restitution_range": [0.0, 1.1]}},
        {"visual": {"color_jitter": 1.1}},
        {"pose": {"tool_x_range": [0.2, -0.2]}},
    ],
)
def test_pullcube_randomization_rejects_invalid_ranges(raw: dict[str, object]) -> None:
    """Invalid physical settings should fail before simulator creation."""

    with pytest.raises(ValueError, match="domain_randomization"):
        parse_pullcube_domain_randomization(raw)
