from __future__ import annotations

"""Isaac Lab task registry and alias resolution.

The registry keeps mini-pi0 configuration names separate from Isaac Lab's Gym
task ids. This lets configs use stable, readable names while the adapter can
resolve the exact backend task id at runtime.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class IsaacLabTaskSpec:
    """Resolved Isaac Lab task metadata.

    Attributes:
        key: Stable mini-pi0 task key.
        gym_id: Isaac Lab Gymnasium task id.
        robot: Canonical robot label expected by the task.
        action_dim: Expected action dimension for the default task action mode.
        state_keys: Recommended canonical state keys.
        image_keys: Recommended canonical image keys.
        notes: Short implementation note for docs/status output.
    """

    key: str
    gym_id: str
    robot: str
    action_dim: int
    state_keys: tuple[str, ...]
    image_keys: tuple[str, ...]
    notes: str


_DEFAULT_STATE_KEYS: tuple[str, ...] = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "observation.state.object",
    "observation.state.task_progress",
)
_DEFAULT_IMAGE_KEYS: tuple[str, ...] = ("agentview_image",)

_TASK_SPECS: dict[str, IsaacLabTaskSpec] = {
    "franka_lift_cube": IsaacLabTaskSpec(
        key="franka_lift_cube",
        gym_id="Isaac-Lift-Cube-Franka-v0",
        robot="franka",
        action_dim=7,
        state_keys=_DEFAULT_STATE_KEYS,
        image_keys=_DEFAULT_IMAGE_KEYS,
        notes="Primary Isaac Lab smoke task for adapter and PPO warm-start validation.",
    ),
    "franka_stack_cube": IsaacLabTaskSpec(
        key="franka_stack_cube",
        gym_id="Isaac-Stack-Cube-Franka-v0",
        robot="franka",
        action_dim=7,
        state_keys=_DEFAULT_STATE_KEYS,
        image_keys=_DEFAULT_IMAGE_KEYS,
        notes="Stacking target kept behind the same adapter contract after lift is validated.",
    ),
    "franka_pick_place": IsaacLabTaskSpec(
        key="franka_pick_place",
        gym_id="Isaac-PickPlace-Franka-v0",
        robot="franka",
        action_dim=7,
        state_keys=_DEFAULT_STATE_KEYS,
        image_keys=_DEFAULT_IMAGE_KEYS,
        notes="Pick/place alias for broader manipulation experiments when the task is available.",
    ),
    "franka_peg_insertion": IsaacLabTaskSpec(
        key="franka_peg_insertion",
        gym_id="Isaac-Peg-Insertion-Franka-v0",
        robot="franka",
        action_dim=7,
        state_keys=_DEFAULT_STATE_KEYS,
        image_keys=_DEFAULT_IMAGE_KEYS,
        notes="Contact-rich target; validate only after lift and stack/pick smoke paths pass.",
    ),
}

_ALIASES: dict[str, str] = {
    "lift": "franka_lift_cube",
    "liftcube": "franka_lift_cube",
    "lift_cube": "franka_lift_cube",
    "franka_lift": "franka_lift_cube",
    "franka_lift_cube": "franka_lift_cube",
    "isaac-lift-cube-franka-v0": "franka_lift_cube",
    "stack": "franka_stack_cube",
    "stackcube": "franka_stack_cube",
    "stack_cube": "franka_stack_cube",
    "franka_stack_cube": "franka_stack_cube",
    "isaac-stack-cube-franka-v0": "franka_stack_cube",
    "pickplace": "franka_pick_place",
    "pick_place": "franka_pick_place",
    "franka_pick_place": "franka_pick_place",
    "isaac-pickplace-franka-v0": "franka_pick_place",
    "peg": "franka_peg_insertion",
    "peginsertion": "franka_peg_insertion",
    "peg_insertion": "franka_peg_insertion",
    "franka_peg_insertion": "franka_peg_insertion",
    "isaac-peg-insertion-franka-v0": "franka_peg_insertion",
}


def list_isaaclab_tasks() -> list[str]:
    """Return supported mini-pi0 Isaac Lab task keys."""

    return sorted(_TASK_SPECS)


def resolve_isaaclab_task(task: str) -> IsaacLabTaskSpec:
    """Resolve a mini-pi0 task key, alias, or direct Isaac Gym id.

    Args:
        task: Configured task name.

    Returns:
        Resolved task metadata.

    Raises:
        ValueError: If the task is unknown.
    """

    raw = str(task or "").strip()
    key = raw.lower().replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key in _TASK_SPECS:
        return _TASK_SPECS[key]
    if raw.startswith("Isaac-") and raw.endswith("-v0"):
        return IsaacLabTaskSpec(
            key=raw,
            gym_id=raw,
            robot="unknown",
            action_dim=7,
            state_keys=_DEFAULT_STATE_KEYS,
            image_keys=_DEFAULT_IMAGE_KEYS,
            notes="Direct Isaac Lab Gym task id supplied by config.",
        )
    options = ", ".join(list_isaaclab_tasks())
    raise ValueError(f"Unknown Isaac Lab task '{task}'. Known mini-pi0 task keys: {options}")
