"""Validated domain-randomization settings for the ManiSkill PullCube task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


Number = int | float
Range = tuple[float, float]


@dataclass(frozen=True)
class PullCubeDomainRandomizationConfig:
    """Randomization ranges used by the CUDA-vectorized PullCube environment."""

    enabled: bool = False
    profile: str = "off"
    seed: int = 0
    robot_init_qpos_noise: float = 0.02
    tool_x_range: Range = (-0.30, -0.10)
    tool_y_range: Range = (-0.30, -0.10)
    cube_x_range: Range = (0.05, 0.25)
    cube_y_range: Range = (-0.25, 0.05)
    cube_yaw_range_deg: Range = (-30.0, 30.0)
    cube_mass_scale_range: Range = (1.0, 1.0)
    tool_mass_scale_range: Range = (1.0, 1.0)
    friction_range: Range = (1.0, 1.0)
    restitution_range: Range = (0.0, 0.0)
    color_jitter: float = 0.0


def parse_pullcube_domain_randomization(
    raw: Mapping[str, object] | None,
) -> PullCubeDomainRandomizationConfig:
    """Parse and validate PullCube domain-randomization settings.

    Args:
        raw: Nested mapping from ``simulator.env_kwargs.domain_randomization``.

    Returns:
        Immutable randomization configuration.

    Raises:
        ValueError: If a range is malformed or physically invalid.
    """

    if raw is None:
        return PullCubeDomainRandomizationConfig()
    pose = _section(raw, "pose")
    physics = _section(raw, "physics")
    visual = _section(raw, "visual")
    config = PullCubeDomainRandomizationConfig(
        enabled=bool(raw.get("enabled", False)),
        profile=str(raw.get("profile", "strong")),
        seed=_integer(raw.get("seed"), 0, "seed"),
        robot_init_qpos_noise=_number(
            raw.get("robot_init_qpos_noise"), 0.02, "robot_init_qpos_noise"
        ),
        tool_x_range=_range(pose.get("tool_x_range"), (-0.30, -0.10), "pose.tool_x_range"),
        tool_y_range=_range(pose.get("tool_y_range"), (-0.30, -0.10), "pose.tool_y_range"),
        cube_x_range=_range(pose.get("cube_x_range"), (0.05, 0.25), "pose.cube_x_range"),
        cube_y_range=_range(pose.get("cube_y_range"), (-0.25, 0.05), "pose.cube_y_range"),
        cube_yaw_range_deg=_range(
            pose.get("cube_yaw_range_deg"), (-30.0, 30.0), "pose.cube_yaw_range_deg"
        ),
        cube_mass_scale_range=_range(
            physics.get("cube_mass_scale_range"), (1.0, 1.0), "physics.cube_mass_scale_range"
        ),
        tool_mass_scale_range=_range(
            physics.get("tool_mass_scale_range"), (1.0, 1.0), "physics.tool_mass_scale_range"
        ),
        friction_range=_range(
            physics.get("friction_range"), (1.0, 1.0), "physics.friction_range"
        ),
        restitution_range=_range(
            physics.get("restitution_range"), (0.0, 0.0), "physics.restitution_range"
        ),
        color_jitter=_number(visual.get("color_jitter"), 0.0, "visual.color_jitter"),
    )
    _validate(config)
    return config


def _section(raw: Mapping[str, object], name: str) -> Mapping[str, object]:
    """Return one optional nested config section."""

    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"domain_randomization.{name} must be a mapping")
    return value


def _number(value: object, default: float, name: str) -> float:
    """Parse one finite numeric field."""

    result = default if value is None else value
    if not isinstance(result, (int, float)):
        raise ValueError(f"domain_randomization.{name} must be numeric")
    parsed = float(result)
    if not -float("inf") < parsed < float("inf"):
        raise ValueError(f"domain_randomization.{name} must be finite")
    return parsed


def _integer(value: object, default: int, name: str) -> int:
    """Parse one integer field without silently truncating floats."""

    result = default if value is None else value
    if not isinstance(result, int) or isinstance(result, bool):
        raise ValueError(f"domain_randomization.{name} must be an integer")
    return result


def _range(value: object, default: Range, name: str) -> Range:
    """Parse one ordered two-value range."""

    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"domain_randomization.{name} must be a [min, max] range")
    lower = _number(value[0], default[0], name)
    upper = _number(value[1], default[1], name)
    if upper < lower:
        raise ValueError(f"domain_randomization.{name} max must be >= min")
    return lower, upper


def _validate(config: PullCubeDomainRandomizationConfig) -> None:
    """Validate physical and visual bounds."""

    if config.robot_init_qpos_noise < 0.0:
        raise ValueError("domain_randomization.robot_init_qpos_noise must be >= 0")
    if min(*config.cube_mass_scale_range, *config.tool_mass_scale_range) <= 0.0:
        raise ValueError("domain_randomization mass scales must be > 0")
    if min(config.friction_range) <= 0.0:
        raise ValueError("domain_randomization friction must be > 0")
    if not (0.0 <= config.restitution_range[0] <= config.restitution_range[1] <= 1.0):
        raise ValueError("domain_randomization restitution must lie in [0, 1]")
    if not 0.0 <= config.color_jitter <= 1.0:
        raise ValueError("domain_randomization visual.color_jitter must lie in [0, 1]")
