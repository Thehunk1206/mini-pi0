from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys
from mini_pi0.sim.registry import make_sim_adapter


@dataclass
class _EpisodeBuffer:
    obs: list[dict[str, np.ndarray]]
    actions: list[np.ndarray]
    rewards: list[float]
    dones: list[int]
    info_rows: list[dict[str, Any]]


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
        self.target_idx = None
        self.phase = "select_target"
        self.phase_step = 0
        self.retry_count = 0
        self.prev_target_pos = None
        self.lift_reference_z = None
        self.open_hold_steps = 0
        self.closed_hold_steps = 0

    def _pick_target(self, obs: dict[str, np.ndarray]) -> int | None:
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
            # Heuristic: partially closed gripper and object moved up from table.
            g_close = bool(np.mean(np.abs(gripper_qpos)) < 0.03)
            lifted = bool(target[2] > 0.04)
            near_eef = bool(np.linalg.norm(target - eef) < 0.09)
            return (g_close and near_eef) or lifted

        # Timeout fallback per phase.
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
                # likely failed grasp -> retry
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
                # If this object is now marked placed, switch target.
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


def _write_episode(group, demo_idx: int, ep: _EpisodeBuffer, final_info: dict[str, Any]) -> None:
    key = f"demo_{demo_idx}"
    if key in group:
        del group[key]
    demo = group.create_group(key)

    t = len(ep.actions)
    demo.attrs["num_samples"] = int(t)
    demo.attrs["success_bool"] = int(bool(final_info.get("success", False)))
    demo.attrs["placed_count"] = int(final_info.get("placed_count", 0))
    demo.attrs["total_objects"] = int(final_info.get("total_objects", 0))
    demo.attrs["final_success_fraction"] = float(final_info.get("success_fraction", 0.0))

    demo.create_dataset("actions", data=np.asarray(ep.actions, dtype=np.float32))
    demo.create_dataset("rewards", data=np.asarray(ep.rewards, dtype=np.float32))

    dones = np.asarray(ep.dones, dtype=np.int32)
    demo.create_dataset("dones", data=dones)

    info_grp = demo.create_group("info")
    for name in (
        "success_fraction",
        "reward_total",
        "reward_progress",
        "reward_place",
        "reward_terminal",
        "reward_shaping",
        "reward_penalties",
        "reward_step",
    ):
        info_grp.create_dataset(
            name,
            data=np.asarray([float(row.get(name, 0.0)) for row in ep.info_rows], dtype=np.float32),
        )

    obs_grp = demo.create_group("obs")
    keys = sorted(ep.obs[0].keys()) if ep.obs else []
    for k in keys:
        arr = np.stack([np.asarray(o[k]) for o in ep.obs], axis=0)
        obs_grp.create_dataset(k, data=arr)


def collect_maniskill_demos(
    cfg: RootConfig,
    *,
    out_hdf5: str,
    num_episodes: int,
    max_steps: int,
    only_success: bool,
    overwrite: bool = False,
) -> dict[str, Any]:
    out_path = Path(out_hdf5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_keys = effective_image_keys(cfg.robot)
    state_keys = effective_state_keys(cfg.robot)

    tray_center = np.asarray(cfg.simulator.env_kwargs.get("tray_center", [0.62, 0.0, 0.0]), dtype=np.float32)
    policy = ScriptedMultiObjectOracle(tray_center=tray_center)

    saved = 0
    trials = 0

    if out_path.exists() and (not overwrite):
        raise FileExistsError(f"Output HDF5 already exists: {out_path}. Pass --overwrite to replace it.")

    with h5py.File(out_path, "w") as h5:
        data_group = h5.require_group("data")
        data_group.attrs["total"] = 0
        data_group.attrs["env_args"] = json.dumps(
            {
                "env_name": cfg.simulator.task,
                "env_type": "maniskill3_custom",
                "env_kwargs": dict(cfg.simulator.env_kwargs),
            }
        )

        while saved < int(num_episodes):
            trials += 1
            ep_cfg = copy.deepcopy(cfg)
            ep_cfg.experiment.seed = int(cfg.experiment.seed + trials - 1)
            adapter = make_sim_adapter(ep_cfg)
            obs = adapter.reset(seed=ep_cfg.experiment.seed)
            policy.reset()

            buf = _EpisodeBuffer(obs=[], actions=[], rewards=[], dones=[], info_rows=[])
            final_info: dict[str, Any] = {"success": False, "success_fraction": 0.0, "placed_count": 0, "total_objects": 0}

            for _ in range(int(max_steps)):
                buf.obs.append({
                    k: np.asarray(obs[k])
                    for k in set(image_keys + state_keys + [
                        "observation.state.object",
                        "observation.state.object_mask",
                        "observation.state.placed_mask",
                        "observation.state.task_progress",
                    ])
                    if k in obs
                })
                action = policy.act(obs)
                step = adapter.step(action)
                buf.actions.append(action.astype(np.float32))
                buf.rewards.append(float(step.reward))
                done = bool(step.done or adapter.check_success(step.info, step.obs))
                buf.dones.append(1 if done else 0)
                buf.info_rows.append(dict(step.info))
                obs = step.obs
                final_info = dict(step.info)
                if done:
                    break

            adapter.close()

            if only_success and not bool(final_info.get("success", False)):
                continue

            _write_episode(data_group, saved, buf, final_info)
            data_group.attrs["total"] = int(data_group.attrs.get("total", 0)) + len(buf.actions)
            saved += 1

    return {
        "path": str(out_path),
        "episodes_saved": int(saved),
        "trials": int(trials),
        "only_success": bool(only_success),
    }
