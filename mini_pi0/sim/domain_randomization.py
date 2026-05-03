"""Domain randomization config parsing for ManiSkill environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence


Number = int | float


@dataclass(frozen=True)
class CameraRandomizationConfig:
    """Camera pose and FOV randomization settings."""

    enabled: bool = False
    base_pos_jitter: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_target_jitter: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hand_pos_jitter: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov_jitter_deg: float = 0.0


@dataclass(frozen=True)
class LightingRandomizationConfig:
    """Lighting randomization settings."""

    enabled: bool = False
    ambient_range: tuple[float, float] = (0.3, 0.3)
    directional_intensity_range: tuple[float, float] = (1.0, 1.0)
    directional_yaw_range_deg: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class VisualRandomizationConfig:
    """Render material color jitter settings."""

    enabled: bool = False
    object_color_jitter: float = 0.0
    tray_color_jitter: float = 0.0
    bowl_color_jitter: float = 0.0
    table_color_jitter: float = 0.0


@dataclass(frozen=True)
class ObjectRandomizationConfig:
    """Object count, slot, and spawn randomization settings."""

    enabled: bool = False
    randomize_active_slots: bool = False
    randomize_spawn_yaw: bool = False
    spawn_radius_jitter: float = 0.0


@dataclass(frozen=True)
class PlacementRandomizationConfig:
    """Tray drop target randomization settings."""

    enabled: bool = False
    target_xy_margin: float = 0.045
    target_z: float = 0.065
    min_target_separation: float = 0.055


@dataclass(frozen=True)
class PhysicsRandomizationConfig:
    """Object physical property randomization settings."""

    enabled: bool = False
    object_mass_scale_range: tuple[float, float] = (1.0, 1.0)
    object_friction_range: tuple[float, float] = (1.0, 1.0)
    object_restitution_range: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Top-level domain randomization settings for the custom ManiSkill task."""

    enabled: bool = False
    profile: str = "off"
    camera: CameraRandomizationConfig = field(default_factory=CameraRandomizationConfig)
    lighting: LightingRandomizationConfig = field(default_factory=LightingRandomizationConfig)
    visual: VisualRandomizationConfig = field(default_factory=VisualRandomizationConfig)
    objects: ObjectRandomizationConfig = field(default_factory=ObjectRandomizationConfig)
    placement: PlacementRandomizationConfig = field(default_factory=PlacementRandomizationConfig)
    physics: PhysicsRandomizationConfig = field(default_factory=PhysicsRandomizationConfig)


def parse_domain_randomization_config(raw: Mapping[str, object] | None) -> DomainRandomizationConfig:
    """Parse nested domain-randomization config from env kwargs.

    Args:
        raw: Raw mapping from YAML/env kwargs, or ``None``.

    Returns:
        Normalized immutable config with validated ranges.

    Raises:
        ValueError: If a vector or range has invalid shape/order.
    """
    if raw is None:
        return DomainRandomizationConfig()
    enabled = _bool(raw.get("enabled", False))
    profile = str(raw.get("profile", "conservative" if enabled else "off"))
    return DomainRandomizationConfig(
        enabled=enabled,
        profile=profile,
        camera=_parse_camera(_section(raw, "camera")),
        lighting=_parse_lighting(_section(raw, "lighting")),
        visual=_parse_visual(_section(raw, "visual")),
        objects=_parse_objects(_section(raw, "objects")),
        placement=_parse_placement(_section(raw, "placement")),
        physics=_parse_physics(_section(raw, "physics")),
    )


def _section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"domain_randomization.{key} must be a mapping")
    return value


def _bool(value: object) -> bool:
    return bool(value)


def _float(value: object, default: float) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric value, got {value!r}")
    return float(value)


def _vec3(value: object, default: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-value sequence")
    return (float(value[0]), float(value[1]), float(value[2]))


def _range(value: object, default: tuple[float, float], name: str) -> tuple[float, float]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a [min, max] range")
    lo = float(value[0])
    hi = float(value[1])
    if hi < lo:
        raise ValueError(f"{name} max must be >= min")
    return (lo, hi)


def _parse_camera(raw: Mapping[str, object]) -> CameraRandomizationConfig:
    return CameraRandomizationConfig(
        enabled=_bool(raw.get("enabled", False)),
        base_pos_jitter=_vec3(raw.get("base_pos_jitter"), (0.0, 0.0, 0.0), "camera.base_pos_jitter"),
        base_target_jitter=_vec3(raw.get("base_target_jitter"), (0.0, 0.0, 0.0), "camera.base_target_jitter"),
        hand_pos_jitter=_vec3(raw.get("hand_pos_jitter"), (0.0, 0.0, 0.0), "camera.hand_pos_jitter"),
        fov_jitter_deg=max(0.0, _float(raw.get("fov_jitter_deg"), 0.0)),
    )


def _parse_lighting(raw: Mapping[str, object]) -> LightingRandomizationConfig:
    return LightingRandomizationConfig(
        enabled=_bool(raw.get("enabled", False)),
        ambient_range=_range(raw.get("ambient_range"), (0.3, 0.3), "lighting.ambient_range"),
        directional_intensity_range=_range(raw.get("directional_intensity_range"), (1.0, 1.0), "lighting.directional_intensity_range"),
        directional_yaw_range_deg=_range(raw.get("directional_yaw_range_deg"), (0.0, 0.0), "lighting.directional_yaw_range_deg"),
    )


def _parse_visual(raw: Mapping[str, object]) -> VisualRandomizationConfig:
    return VisualRandomizationConfig(
        enabled=_bool(raw.get("enabled", False)),
        object_color_jitter=max(0.0, _float(raw.get("object_color_jitter"), 0.0)),
        tray_color_jitter=max(0.0, _float(raw.get("tray_color_jitter"), 0.0)),
        bowl_color_jitter=max(0.0, _float(raw.get("bowl_color_jitter"), 0.0)),
        table_color_jitter=max(0.0, _float(raw.get("table_color_jitter"), 0.0)),
    )


def _parse_objects(raw: Mapping[str, object]) -> ObjectRandomizationConfig:
    return ObjectRandomizationConfig(
        enabled=_bool(raw.get("enabled", False)),
        randomize_active_slots=_bool(raw.get("randomize_active_slots", False)),
        randomize_spawn_yaw=_bool(raw.get("randomize_spawn_yaw", False)),
        spawn_radius_jitter=max(0.0, _float(raw.get("spawn_radius_jitter"), 0.0)),
    )


def _parse_placement(raw: Mapping[str, object]) -> PlacementRandomizationConfig:
    return PlacementRandomizationConfig(
        enabled=_bool(raw.get("enabled", False)),
        target_xy_margin=max(0.0, _float(raw.get("target_xy_margin"), 0.045)),
        target_z=max(0.0, _float(raw.get("target_z"), 0.065)),
        min_target_separation=max(0.0, _float(raw.get("min_target_separation"), 0.055)),
    )


def _parse_physics(raw: Mapping[str, object]) -> PhysicsRandomizationConfig:
    return PhysicsRandomizationConfig(
        enabled=_bool(raw.get("enabled", False)),
        object_mass_scale_range=_range(raw.get("object_mass_scale_range"), (1.0, 1.0), "physics.object_mass_scale_range"),
        object_friction_range=_range(raw.get("object_friction_range"), (1.0, 1.0), "physics.object_friction_range"),
        object_restitution_range=_range(raw.get("object_restitution_range"), (0.0, 0.0), "physics.object_restitution_range"),
    )
