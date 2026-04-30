"""Scripted oracle policy used by ManiSkill collector backends."""

from __future__ import annotations

import numpy as np


class ScriptedMultiObjectOracle:
    """Waypoint + condition-checked finite-state oracle for pick-place.

    Action assumes EE-delta style control where first 3 dimensions are Cartesian
    deltas and last dimension controls gripper open/close.
    """

    def __init__(self, tray_center: np.ndarray):
        self.tray_center = tray_center.astype(np.float32)
        self.target_idx = None
        self.phase = "select_target"
        self.phase_step = 0
        self.retry_count = 0
        self.prev_target_pos = None
        self.lift_reference_z = None
        self.open_hold_steps = 0
        self.closed_hold_steps = 0

    def reset(self) -> None:
        """Reset finite-state controller internals for a fresh episode.

        Clears target selection, phase machine counters, and retry bookkeeping.
        """
        self.target_idx = None
        self.phase = "select_target"
        self.phase_step = 0
        self.retry_count = 0
        self.prev_target_pos = None
        self.lift_reference_z = None
        self.open_hold_steps = 0
        self.closed_hold_steps = 0

    def _pick_target(self, obs: dict[str, np.ndarray]) -> int | None:
        """Select nearest active, not-yet-placed object index.

        Args:
            obs: Canonical observation dict.

        Returns:
            Object index to target next, or `None` if no candidate exists.
        """
        obj = np.asarray(obs.get("observation.state.object"), dtype=np.float32).reshape(-1, 3)
        obj_mask = np.asarray(obs.get("observation.state.object_mask"), dtype=np.float32).reshape(-1)
        placed = np.asarray(obs.get("observation.state.placed_mask"), dtype=np.float32).reshape(-1)
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)

        best_idx = None
        best_dist = float("inf")
        for i in range(min(len(obj), len(obj_mask), len(placed))):
            if obj_mask[i] < 0.5 or placed[i] > 0.5:
                continue
            d = float(np.linalg.norm(eef - obj[i]))
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Generate one normalized action in `[-1, 1]` for current phase.

        Args:
            obs: Canonical observation dict for current step.

        Returns:
            Action vector shaped for EE-delta controller (`7` dims).
        """
        action = np.zeros((7,), dtype=np.float32)
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        obj = np.asarray(obs.get("observation.state.object"), dtype=np.float32).reshape(-1, 3)
        placed = np.asarray(obs.get("observation.state.placed_mask"), dtype=np.float32).reshape(-1)
        gripper_qpos = np.asarray(obs.get("robot0_gripper_qpos", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)

        self.phase_step += 1

        if self.target_idx is None or self.target_idx >= len(placed) or placed[self.target_idx] > 0.5:
            self.target_idx = self._pick_target(obs)
            self.phase = "select_target"
            self.phase_step = 0
            self.retry_count = 0

        if self.target_idx is None:
            action[6] = 1.0
            return action

        target = obj[self.target_idx]
        above = target + np.array([0.0, 0.0, 0.11], dtype=np.float32)
        pre_grasp = target + np.array([0.0, 0.0, 0.055], dtype=np.float32)
        grasp = target + np.array([0.0, 0.0, 0.022], dtype=np.float32)
        lift_goal = target + np.array([0.0, 0.0, 0.15], dtype=np.float32)
        tray_above = self.tray_center + np.array([0.0, 0.0, 0.16], dtype=np.float32)
        tray_drop = self.tray_center + np.array([0.0, 0.0, 0.065], dtype=np.float32)
        retreat = self.tray_center + np.array([0.0, 0.0, 0.18], dtype=np.float32)

        def delta(goal: np.ndarray, gain: float = 10.0) -> np.ndarray:
            d = (goal - eef) * gain
            d = np.clip(d, -1.0, 1.0)
            return d.astype(np.float32)

        def transition(next_phase: str) -> None:
            self.phase = next_phase
            self.phase_step = 0

        def likely_grasped() -> bool:
            g_close = bool(np.mean(np.abs(gripper_qpos)) < 0.03)
            lifted = bool(target[2] > 0.04)
            near_eef = bool(np.linalg.norm(target - eef) < 0.09)
            return (g_close and near_eef) or lifted

        if self.phase_step > 90:
            self.retry_count += 1
            transition("select_target")
            if self.retry_count > 4:
                self.target_idx = None
                self.retry_count = 0

        if self.phase == "select_target":
            self.prev_target_pos = target.copy()
            self.lift_reference_z = float(target[2])
            self.open_hold_steps = 0
            self.closed_hold_steps = 0
            transition("approach_above")
            action[6] = 1.0
            return action

        if self.phase == "approach_above":
            action[:3] = delta(above)
            action[6] = 1.0
            if np.linalg.norm(above - eef) < 0.022:
                transition("pre_grasp")
        elif self.phase == "pre_grasp":
            action[:3] = delta(pre_grasp)
            action[6] = 1.0
            if np.linalg.norm(pre_grasp - eef) < 0.018:
                transition("descend_grasp")
        elif self.phase == "descend_grasp":
            action[:3] = delta(grasp)
            action[6] = 1.0
            if np.linalg.norm(grasp - eef) < 0.012:
                transition("close_gripper")
        elif self.phase == "close_gripper":
            action[:3] = 0.0
            action[6] = -1.0
            self.closed_hold_steps += 1
            if self.closed_hold_steps >= 8:
                transition("lift")
        elif self.phase == "lift":
            action[:3] = delta(lift_goal)
            action[6] = -1.0
            if likely_grasped() and eef[2] > (self.lift_reference_z + 0.08):
                transition("to_tray_above")
            elif self.phase_step > 45:
                self.retry_count += 1
                transition("select_target")
        elif self.phase == "to_tray_above":
            action[:3] = delta(tray_above)
            action[6] = -1.0
            if np.linalg.norm(tray_above - eef) < 0.026:
                transition("drop_to_tray")
        elif self.phase == "drop_to_tray":
            action[:3] = delta(tray_drop)
            action[6] = -1.0
            if np.linalg.norm(tray_drop - eef) < 0.016:
                transition("open_gripper")
        elif self.phase == "open_gripper":
            action[:3] = 0.0
            action[6] = 1.0
            self.open_hold_steps += 1
            if self.open_hold_steps >= 10:
                transition("retreat")
        elif self.phase == "retreat":
            action[:3] = delta(retreat)
            action[6] = 1.0
            if np.linalg.norm(retreat - eef) < 0.025:
                if self.target_idx < len(placed) and placed[self.target_idx] > 0.5:
                    self.target_idx = None
                    self.retry_count = 0
                else:
                    self.retry_count += 1
                self.target_idx = None
                transition("select_target")
        else:
            transition("select_target")
            action[6] = 1.0

        return np.clip(action, -1.0, 1.0)
