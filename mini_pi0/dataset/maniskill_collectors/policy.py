"""Scripted oracle policy used by ManiSkill collector backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class OracleProfile(str, Enum):
    """Supported scripted oracle dataset collection profiles."""

    CORE = "core"
    RECOVERY = "recovery"
    SUBOPTIMAL = "suboptimal"


@dataclass(frozen=True)
class OracleOptions:
    """Runtime options used to diversify scripted oracle trajectories."""

    profile: OracleProfile = OracleProfile.CORE
    action_noise_std: float = 0.0
    action_noise_clip: float = 0.0
    speed_scale: float = 1.0
    grasp_pose_noise_xy: float = 0.0
    grasp_pose_noise_z: float = 0.0
    grasp_angle_jitter_deg: float = 0.0
    allow_regrasp: bool = True


class ScriptedMultiObjectOracle:
    """Waypoint + condition-checked finite-state oracle for pick-place.

    Action assumes EE-delta style control where first 3 dimensions are Cartesian
    deltas and last dimension controls gripper open/close.
    """

    def __init__(self, tray_center: np.ndarray, options: OracleOptions | None = None, rng: np.random.Generator | None = None):
        self.tray_center = tray_center.astype(np.float32)
        self.options = options or OracleOptions()
        self.rng = rng or np.random.default_rng()
        self.target_idx = None
        self.phase = "select_target"
        self.phase_step = 0
        self.retry_count = 0
        self.retry_count_total = 0
        self.phase_timeout_count = 0
        self.target_switch_count = 0
        self.max_phase_steps = 0
        self.prev_target_pos = None
        self.lift_reference_z = None
        self.open_hold_steps = 0
        self.closed_hold_steps = 0
        self._grasp_noise = np.zeros((3,), dtype=np.float32)
        self._yaw_jitter = 0.0

    def reset(self) -> None:
        """Reset finite-state controller internals for a fresh episode.

        Clears target selection, phase machine counters, and retry bookkeeping.
        """
        self.target_idx = None
        self.phase = "select_target"
        self.phase_step = 0
        self.retry_count = 0
        self.retry_count_total = 0
        self.phase_timeout_count = 0
        self.target_switch_count = 0
        self.max_phase_steps = 0
        self.prev_target_pos = None
        self.lift_reference_z = None
        self.open_hold_steps = 0
        self.closed_hold_steps = 0
        self._grasp_noise = np.zeros((3,), dtype=np.float32)
        self._yaw_jitter = 0.0

    def telemetry(self) -> dict[str, int]:
        """Return quality-filter telemetry for the current rollout."""
        return {
            "oracle_retry_count": int(self.retry_count_total),
            "oracle_phase_timeout_count": int(self.phase_timeout_count),
            "oracle_target_switch_count": int(self.target_switch_count),
            "oracle_max_phase_steps": int(self.max_phase_steps),
        }

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

    @staticmethod
    def _wrap_symmetric_yaw_error(error: float) -> float:
        """Wrap yaw error for a parallel-jaw gripper's 180-degree symmetry."""
        while error > np.pi / 2:
            error -= np.pi
        while error < -np.pi / 2:
            error += np.pi
        return float(error)

    @staticmethod
    def _closing_axis_xy(eef_quat_wxyz: np.ndarray) -> np.ndarray:
        """Return the gripper closing axis projected into the table plane."""
        if eef_quat_wxyz.shape[0] < 4:
            return np.array([1.0, 0.0], dtype=np.float32)
        w, x, y, z = [float(v) for v in eef_quat_wxyz[:4]]
        axis = np.array(
            [
                2.0 * (x * y - z * w),
                1.0 - 2.0 * (x * x + z * z),
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32)
        return axis / norm

    @staticmethod
    def _desired_closing_axis_xy(target_idx: int, obj: np.ndarray, obj_mask: np.ndarray, placed: np.ndarray) -> np.ndarray:
        """Choose a gripper closing axis that avoids nearby clutter."""
        target_xy = obj[target_idx, :2]
        best_idx = None
        best_dist = float("inf")
        for idx in range(min(len(obj), len(obj_mask), len(placed))):
            if idx == target_idx or obj_mask[idx] < 0.5 or placed[idx] > 0.5:
                continue
            dist = float(np.linalg.norm(obj[idx, :2] - target_xy))
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is None or best_dist > 0.09:
            return np.array([1.0, 0.0], dtype=np.float32)

        neighbor_vec = obj[best_idx, :2] - target_xy
        norm = float(np.linalg.norm(neighbor_vec))
        if norm < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32)
        neighbor_axis = neighbor_vec / norm
        return np.array([-neighbor_axis[1], neighbor_axis[0]], dtype=np.float32)

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Generate one normalized action in `[-1, 1]` for current phase.

        Args:
            obs: Canonical observation dict for current step.

        Returns:
            Action vector shaped for EE-delta controller (`7` dims).
        """
        action = np.zeros((7,), dtype=np.float32)
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        eef_quat = np.asarray(
            obs.get("robot0_eef_quat", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        obj = np.asarray(obs.get("observation.state.object"), dtype=np.float32).reshape(-1, 3)
        obj_mask = np.asarray(obs.get("observation.state.object_mask"), dtype=np.float32).reshape(-1)
        placed = np.asarray(obs.get("observation.state.placed_mask"), dtype=np.float32).reshape(-1)
        place_targets = np.asarray(obs.get("observation.state.place_targets", np.array([], dtype=np.float32)), dtype=np.float32)
        grasped_mask = np.asarray(obs.get("observation.state.grasped_mask", np.zeros_like(placed)), dtype=np.float32).reshape(-1)
        gripper_qpos = np.asarray(obs.get("robot0_gripper_qpos", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)

        self.phase_step += 1
        self.max_phase_steps = max(self.max_phase_steps, int(self.phase_step))

        if self.target_idx is None or self.target_idx >= len(placed) or placed[self.target_idx] > 0.5:
            prev_target_idx = self.target_idx
            self.target_idx = self._pick_target(obs)
            if prev_target_idx is not None and self.target_idx is not None and self.target_idx != prev_target_idx:
                self.target_switch_count += 1
            self.phase = "select_target"
            self.phase_step = 0
            self.retry_count = 0

        if self.target_idx is None:
            action[6] = 1.0
            return action

        target = obj[self.target_idx]
        target_for_grasp = target + self._grasp_noise
        if place_targets.size >= obj.size:
            drop_target = place_targets.reshape(-1, 3)[self.target_idx].astype(np.float32)
        else:
            drop_target = self.tray_center + np.array([0.0, 0.0, 0.065], dtype=np.float32)
        current_closing = self._closing_axis_xy(eef_quat)
        desired_closing = self._desired_closing_axis_xy(self.target_idx, obj, obj_mask, placed)
        yaw_error = np.arctan2(
            current_closing[0] * desired_closing[1] - current_closing[1] * desired_closing[0],
            float(np.dot(current_closing, desired_closing)),
        )
        yaw_error = self._wrap_symmetric_yaw_error(float(yaw_error))
        yaw_error += self._yaw_jitter
        # Positive Z action produced negative world yaw in the Panda controller
        # probe, so this sign compensates that mapping.
        yaw_action = float(np.clip(-4.0 * yaw_error, -1.0, 1.0))
        above = target_for_grasp + np.array([0.0, 0.0, 0.11], dtype=np.float32)
        pre_grasp = target_for_grasp + np.array([0.0, 0.0, 0.045], dtype=np.float32)
        grasp = target_for_grasp + np.array([0.0, 0.0, 0.004], dtype=np.float32)
        lift_goal = target + np.array([0.0, 0.0, 0.15], dtype=np.float32)
        tray_above = drop_target + np.array([0.0, 0.0, 0.095], dtype=np.float32)
        tray_drop = drop_target
        retreat = drop_target + np.array([0.0, 0.0, 0.115], dtype=np.float32)

        def delta(goal: np.ndarray, gain: float = 4.0) -> np.ndarray:
            d = (goal - eef) * gain * float(max(0.1, self.options.speed_scale))
            d = np.clip(d, -1.0, 1.0)
            return d.astype(np.float32)

        def transition(next_phase: str) -> None:
            self.phase = next_phase
            self.phase_step = 0

        def likely_grasped() -> bool:
            grasped = bool(self.target_idx < len(grasped_mask) and grasped_mask[self.target_idx] > 0.5)
            lifted = bool(target[2] > 0.04)
            return grasped or lifted

        if self.phase_step > 90:
            self.retry_count += 1
            self.retry_count_total += 1
            self.phase_timeout_count += 1
            transition("select_target")
            if self.retry_count > 4:
                self.target_idx = None
                self.retry_count = 0

        if self.phase == "select_target":
            self.prev_target_pos = target.copy()
            self.lift_reference_z = float(target[2])
            self.open_hold_steps = 0
            self.closed_hold_steps = 0
            self._grasp_noise = np.array(
                [
                    self.rng.uniform(-self.options.grasp_pose_noise_xy, self.options.grasp_pose_noise_xy),
                    self.rng.uniform(-self.options.grasp_pose_noise_xy, self.options.grasp_pose_noise_xy),
                    self.rng.uniform(-self.options.grasp_pose_noise_z, self.options.grasp_pose_noise_z),
                ],
                dtype=np.float32,
            )
            self._yaw_jitter = float(np.deg2rad(self.rng.uniform(-self.options.grasp_angle_jitter_deg, self.options.grasp_angle_jitter_deg)))
            transition("approach_above")
            action[6] = 1.0
            return action

        if self.phase == "approach_above":
            action[:3] = delta(above, gain=5.0)
            action[5] = yaw_action
            action[6] = 1.0
            if np.linalg.norm(above - eef) < 0.018:
                transition("pre_grasp")
        elif self.phase == "pre_grasp":
            action[:3] = delta(pre_grasp, gain=4.5)
            action[5] = yaw_action
            action[6] = 1.0
            if np.linalg.norm(pre_grasp - eef) < 0.014:
                transition("descend_grasp")
        elif self.phase == "descend_grasp":
            action[:3] = delta(grasp, gain=3.5)
            action[5] = yaw_action
            action[6] = 1.0
            if np.linalg.norm(grasp - eef) < 0.010:
                transition("close_gripper")
        elif self.phase == "close_gripper":
            action[:3] = 0.0
            action[6] = -1.0
            self.closed_hold_steps += 1
            if self.closed_hold_steps >= 14:
                transition("lift")
        elif self.phase == "lift":
            action[:3] = delta(lift_goal, gain=3.5)
            action[6] = -1.0
            if likely_grasped() and eef[2] > (self.lift_reference_z + 0.08):
                transition("to_tray_above")
            elif self.phase_step > 45:
                self.retry_count += 1
                self.retry_count_total += 1
                transition("select_target")
        elif self.phase == "to_tray_above":
            action[:3] = delta(tray_above, gain=3.0)
            action[6] = -1.0
            if np.linalg.norm(tray_above - eef) < 0.08:
                transition("drop_to_tray")
        elif self.phase == "drop_to_tray":
            action[:3] = delta(tray_drop, gain=2.5)
            action[6] = -1.0
            if np.linalg.norm(tray_drop - eef) < 0.06:
                transition("open_gripper")
        elif self.phase == "open_gripper":
            action[:3] = 0.0
            action[6] = 1.0
            self.open_hold_steps += 1
            if self.open_hold_steps >= 10:
                transition("retreat")
        elif self.phase == "retreat":
            action[:3] = delta(retreat, gain=3.0)
            action[6] = 1.0
            if np.linalg.norm(retreat - eef) < 0.08:
                if self.target_idx < len(placed) and placed[self.target_idx] > 0.5:
                    self.target_idx = None
                    self.retry_count = 0
                else:
                    self.retry_count += 1
                    self.retry_count_total += 1
                self.target_idx = None
                transition("select_target")
        else:
            transition("select_target")
            action[6] = 1.0

        if self.options.action_noise_std > 0.0:
            noise = self.rng.normal(0.0, self.options.action_noise_std, size=action.shape).astype(np.float32)
            if self.options.action_noise_clip > 0.0:
                noise = np.clip(noise, -self.options.action_noise_clip, self.options.action_noise_clip)
            noise[6] = 0.0
            action = action + noise

        return np.clip(action, -1.0, 1.0)
