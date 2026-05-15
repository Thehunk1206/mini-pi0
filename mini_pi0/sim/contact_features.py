"""Compact contact and joint-force features for ManiSkill environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch


DEFAULT_CONTACT_LINKS: Final[tuple[str, ...]] = (
    "panda_leftfinger",
    "panda_rightfinger",
    "panda_hand",
    "panda_hand_tcp",
    "panda_link8",
)
DEFAULT_CONTACT_OBJECTS: Final[tuple[str, ...]] = ("peg", "box")


@dataclass(frozen=True)
class ContactTarget:
    """One body or actor whose raw PhysX contact impulse is tracked."""

    key: str
    aliases: tuple[str, ...]

    def matches(self, body_name: str) -> bool:
        """Return whether a raw contact body belongs to this target."""

        normalized = body_name.lower()
        return any(alias.lower() in normalized for alias in self.aliases)


def collect_contact_features(
    base_env: object,
    *,
    link_names: tuple[str, ...] = DEFAULT_CONTACT_LINKS,
    object_names: tuple[str, ...] = DEFAULT_CONTACT_OBJECTS,
    contact_threshold: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Collect compact contact and joint-force features from one ManiSkill env.

    Args:
        base_env: Unwrapped ManiSkill environment. This implementation targets
            single-environment CPU simulation.
        link_names: Robot link names to track.
        object_names: Task actor attributes to track, e.g. ``peg`` and ``box``.
        contact_threshold: Force norm threshold used for binary contact flags.

    Returns:
        Mapping from feature name to one unbatched float32 vector.
    """

    robot = base_env.agent.robot
    values = _joint_force_features(robot)
    targets = _build_targets(base_env, robot, link_names=link_names, object_names=object_names)
    raw = _raw_contact_summary(base_env=base_env, targets=targets)
    timestep = float(getattr(base_env.scene, "timestep", 1.0))

    for target in targets:
        impulse = raw.target_impulses[target.key]
        force = impulse / max(timestep, 1e-8)
        norm = np.asarray([np.linalg.norm(force)], dtype=np.float32)
        values[f"{target.key}_force"] = force.astype(np.float32)
        values[f"{target.key}_force_norm"] = norm
        values[f"{target.key}_contact"] = (norm > float(contact_threshold)).astype(np.float32)
        values[f"{target.key}_contact_count"] = np.asarray([raw.target_counts[target.key]], dtype=np.float32)

    for pair_key, impulse in raw.pair_impulses.items():
        force = impulse / max(timestep, 1e-8)
        norm = np.asarray([np.linalg.norm(force)], dtype=np.float32)
        values[f"pair_{pair_key}_force"] = force.astype(np.float32)
        values[f"pair_{pair_key}_force_norm"] = norm
        values[f"pair_{pair_key}_contact"] = (norm > float(contact_threshold)).astype(np.float32)
        values[f"pair_{pair_key}_contact_count"] = np.asarray([raw.pair_counts[pair_key]], dtype=np.float32)

    return values


@dataclass(frozen=True)
class RawContactSummary:
    """Aggregated raw contact impulses and counts."""

    target_impulses: dict[str, np.ndarray]
    target_counts: dict[str, float]
    pair_impulses: dict[str, np.ndarray]
    pair_counts: dict[str, float]


def _joint_force_features(robot: object) -> dict[str, np.ndarray]:
    robot_qf = _call_first_env_array(robot, "get_qf", fallback_size=9)
    passive_qf = _call_first_env_array(robot, "compute_passive_force", fallback_size=robot_qf.shape[0])
    return {
        "robot_qf": robot_qf,
        "robot_arm_qf": robot_qf[:7],
        "robot_gripper_qf": robot_qf[7:],
        "robot_passive_qf": passive_qf,
        "robot_arm_passive_qf": passive_qf[:7],
        "robot_gripper_passive_qf": passive_qf[7:],
    }


def _build_targets(
    base_env: object,
    robot: object,
    *,
    link_names: tuple[str, ...],
    object_names: tuple[str, ...],
) -> tuple[ContactTarget, ...]:
    targets: list[ContactTarget] = []
    links_map = getattr(robot, "links_map", {})
    for link_name in link_names:
        link = links_map.get(link_name)
        if link is None:
            continue
        key = _signal_prefix(link_name)
        aliases = {link_name, key, _entity_name(link)}
        targets.append(ContactTarget(key=key, aliases=tuple(alias for alias in aliases if alias)))

    for object_name in object_names:
        actor = getattr(base_env, object_name, None)
        if actor is None:
            continue
        key = _signal_prefix(object_name)
        aliases = {object_name, key, str(getattr(actor, "name", "")), _entity_name(actor)}
        targets.append(ContactTarget(key=key, aliases=tuple(alias for alias in aliases if alias)))
    return tuple(targets)


def _raw_contact_summary(base_env: object, targets: tuple[ContactTarget, ...]) -> RawContactSummary:
    target_impulses = {target.key: np.zeros(3, dtype=np.float32) for target in targets}
    target_counts = {target.key: 0.0 for target in targets}
    target_pairs = _target_pairs(targets)
    pair_impulses = {f"{left.key}_{right.key}": np.zeros(3, dtype=np.float32) for left, right in target_pairs}
    pair_counts = {key: 0.0 for key in pair_impulses}

    px = getattr(base_env.scene, "px", None)
    if px is None or not hasattr(px, "get_contacts"):
        return RawContactSummary(target_impulses, target_counts, pair_impulses, pair_counts)

    for contact in px.get_contacts():
        body0 = _contact_body_name(contact, 0)
        body1 = _contact_body_name(contact, 1)
        impulse = _contact_impulse(contact)
        for target in targets:
            if target.matches(body0):
                target_impulses[target.key] += impulse
                target_counts[target.key] += 1.0
            if target.matches(body1):
                target_impulses[target.key] -= impulse
                target_counts[target.key] += 1.0
        for left, right in target_pairs:
            pair_key = f"{left.key}_{right.key}"
            if left.matches(body0) and right.matches(body1):
                pair_impulses[pair_key] += impulse
                pair_counts[pair_key] += 1.0
            elif left.matches(body1) and right.matches(body0):
                pair_impulses[pair_key] -= impulse
                pair_counts[pair_key] += 1.0
    return RawContactSummary(target_impulses, target_counts, pair_impulses, pair_counts)


def _target_pairs(targets: tuple[ContactTarget, ...]) -> tuple[tuple[ContactTarget, ContactTarget], ...]:
    pairs: list[tuple[ContactTarget, ContactTarget]] = []
    for left_idx, left in enumerate(targets):
        for right in targets[left_idx + 1 :]:
            pairs.append((left, right))
    return tuple(pairs)


def _call_first_env_array(obj: object, method_name: str, fallback_size: int) -> np.ndarray:
    method = getattr(obj, method_name, None)
    if method is None:
        return np.zeros((fallback_size,), dtype=np.float32)
    try:
        return _first_env_array(method())
    except Exception:
        return np.zeros((fallback_size,), dtype=np.float32)


def _first_env_array(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim >= 2 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array, dtype=np.float32).reshape(-1)


def _contact_body_name(contact: object, index: int) -> str:
    body = contact.bodies[index]
    entity = getattr(body, "entity", None)
    return str(getattr(entity, "name", ""))


def _contact_impulse(contact: object) -> np.ndarray:
    if not getattr(contact, "points", None):
        return np.zeros(3, dtype=np.float32)
    return np.sum([point.impulse for point in contact.points], axis=0).astype(np.float32)


def _entity_name(obj: object) -> str:
    bodies = getattr(obj, "_bodies", None)
    if bodies:
        entity = getattr(bodies[0], "entity", None)
        return str(getattr(entity, "name", ""))
    return str(getattr(obj, "name", ""))


def _signal_prefix(link_name: str) -> str:
    return str(link_name).replace("panda_", "").replace("-", "_")
