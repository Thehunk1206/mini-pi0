from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import gymnasium as gym
import numpy as np
import sapien
import torch

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
    demo.attrs["collector_type"] = str(final_info.get("collector_type", "scripted"))

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
    collector_backend: str = "scripted",
    num_envs: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    out_path = Path(out_hdf5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_keys = effective_image_keys(cfg.robot)
    state_keys = effective_state_keys(cfg.robot)

    tray_center = np.asarray(cfg.simulator.env_kwargs.get("tray_center", [0.62, 0.0, 0.0]), dtype=np.float32)
    policy = ScriptedMultiObjectOracle(tray_center=tray_center)
    use_mplib_backend = bool(collector_backend == "mplib")
    num_envs = int(max(1, num_envs))
    if use_mplib_backend and (not _mplib_runtime_check()):
        use_mplib_backend = False

    saved = 0
    trials = 0

    if out_path.exists() and (not overwrite):
        raise FileExistsError(f"Output HDF5 already exists: {out_path}. Pass --overwrite to replace it.")

    collected_episode_stats: list[dict[str, Any]] = []

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
            if trials > max(10, int(num_episodes) * 5):
                break
            ep_cfg = copy.deepcopy(cfg)
            ep_cfg.experiment.seed = int(cfg.experiment.seed + trials - 1)
            if num_envs > 1:
                if use_mplib_backend:
                    out = _collect_vectorized_mplib_episodes(
                        ep_cfg,
                        image_keys=image_keys,
                        state_keys=state_keys,
                        num_envs=num_envs,
                        episodes_target=int(num_episodes) - int(saved),
                        max_steps=int(max_steps),
                        only_success=bool(only_success),
                    )
                else:
                    out = _collect_vectorized_scripted_episodes(
                        ep_cfg,
                        image_keys=image_keys,
                        state_keys=state_keys,
                        num_envs=num_envs,
                        episodes_target=int(num_episodes) - int(saved),
                        max_steps=int(max_steps),
                        only_success=bool(only_success),
                    )
                for ep, final_info in out:
                    final_info["collector_type"] = "mplib" if use_mplib_backend else "scripted"
                    _write_episode(data_group, saved, ep, final_info)
                    data_group.attrs["total"] = int(data_group.attrs.get("total", 0)) + len(ep.actions)
                    collected_episode_stats.append(
                        {
                            "num_samples": len(ep.actions),
                            "success_bool": int(bool(final_info.get("success", False))),
                            "final_success_fraction": float(final_info.get("success_fraction", 0.0)),
                            "placed_count": int(final_info.get("placed_count", 0)),
                            "total_objects": int(final_info.get("total_objects", 0)),
                        }
                    )
                    saved += 1
                continue
            elif use_mplib_backend:
                buf, final_info = _collect_single_episode_mplib(ep_cfg, max_steps=max_steps, image_keys=image_keys, state_keys=state_keys)
            else:
                adapter = make_sim_adapter(ep_cfg)
                obs = adapter.reset(seed=ep_cfg.experiment.seed)
                policy.reset()

                buf = _EpisodeBuffer(obs=[], actions=[], rewards=[], dones=[], info_rows=[])
                final_info = {"success": False, "success_fraction": 0.0, "placed_count": 0, "total_objects": 0}

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

            final_info["collector_type"] = "mplib" if use_mplib_backend else "scripted"

            if only_success and not bool(final_info.get("success", False)):
                continue

            _write_episode(data_group, saved, buf, final_info)
            data_group.attrs["total"] = int(data_group.attrs.get("total", 0)) + len(buf.actions)
            collected_episode_stats.append(
                {
                    "num_samples": len(buf.actions),
                    "success_bool": int(bool(final_info.get("success", False))),
                    "final_success_fraction": float(final_info.get("success_fraction", 0.0)),
                    "placed_count": int(final_info.get("placed_count", 0)),
                    "total_objects": int(final_info.get("total_objects", 0)),
                }
            )
            saved += 1

    stats_summary = _summarize_collection_stats(collected_episode_stats)

    return {
        "path": str(out_path),
        "episodes_saved": int(saved),
        "trials": int(trials),
        "only_success": bool(only_success),
        "collector_backend": "mplib" if use_mplib_backend else "scripted",
        "num_envs": int(num_envs),
        "stats": stats_summary,
    }


def _summarize_collection_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "partial_rate": 0.0,
            "fail_rate": 0.0,
            "mean_success_fraction": 0.0,
            "episode_len_mean": 0.0,
            "episode_len_min": 0,
            "episode_len_max": 0,
            "object_count_histogram": {},
        }
    lens = np.asarray([int(r["num_samples"]) for r in rows], dtype=np.int32)
    succ = np.asarray([int(r["success_bool"]) for r in rows], dtype=np.int32)
    frac = np.asarray([float(r["final_success_fraction"]) for r in rows], dtype=np.float32)
    total_obj = np.asarray([int(r["total_objects"]) for r in rows], dtype=np.int32)
    partial = (frac > 1e-6) & (frac < 1.0 - 1e-6)
    fail = frac <= 1e-6
    hist: dict[str, int] = {}
    for v in total_obj.tolist():
        k = str(int(v))
        hist[k] = hist.get(k, 0) + 1
    return {
        "episodes": int(len(rows)),
        "success_rate": float(np.mean(succ)),
        "partial_rate": float(np.mean(partial.astype(np.float32))),
        "fail_rate": float(np.mean(fail.astype(np.float32))),
        "mean_success_fraction": float(np.mean(frac)),
        "episode_len_mean": float(np.mean(lens)),
        "episode_len_min": int(np.min(lens)),
        "episode_len_max": int(np.max(lens)),
        "object_count_histogram": hist,
    }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _canonical_obs_batch_from_raw_env(env, image_keys: list[str], state_keys: list[str], raw_obs: Any) -> list[dict[str, np.ndarray]]:
    uw = env.unwrapped
    num_envs = int(uw.num_envs)
    tcp_p = _to_numpy(uw.agent.tcp.pose.p).astype(np.float32)
    tcp_q = _to_numpy(uw.agent.tcp.pose.q).astype(np.float32)
    qpos = _to_numpy(uw.agent.robot.get_qpos()).astype(np.float32)
    obj_pos = _to_numpy(uw._get_object_pos_tensor()).reshape(num_envs, -1).astype(np.float32)
    active = _to_numpy(uw._active_object_mask).astype(np.float32)
    placed = _to_numpy(uw._placed_mask).astype(np.float32)
    frac = _to_numpy(uw._last_success_fraction).astype(np.float32)

    frame = None
    if isinstance(raw_obs, dict):
        try:
            rgb = raw_obs["sensor_data"]["base_camera"]["rgb"]
            arr = _to_numpy(rgb)
            if arr.ndim == 3:
                arr = arr[None, ...]
            frame = np.clip(arr, 0, 255).astype(np.uint8)
        except Exception:
            frame = None
    if frame is None:
        h = 128
        w = 128
        frame = np.zeros((num_envs, h, w, 3), dtype=np.uint8)

    out_batch: list[dict[str, np.ndarray]] = []
    for i in range(num_envs):
        default_state = {
            "robot0_eef_pos": tcp_p[i],
            "observation.state.eef_pos": tcp_p[i],
            "robot0_eef_quat": tcp_q[i],
            "observation.state.eef_quat": tcp_q[i],
            "robot0_gripper_qpos": qpos[i][-2:] if qpos.shape[1] >= 2 else qpos[i],
            "observation.state.tool": qpos[i][-2:] if qpos.shape[1] >= 2 else qpos[i],
            "observation.state.object": obj_pos[i],
            "observation.state.object_mask": active[i],
            "observation.state.placed_mask": placed[i],
            "observation.state.task_progress": np.array([frac[i]], dtype=np.float32),
        }
        obs_i: dict[str, np.ndarray] = {}
        for key in image_keys:
            obs_i[key] = frame[i][..., :3]
        for key in state_keys:
            obs_i[key] = np.asarray(default_state.get(key, np.zeros((1,), dtype=np.float32)), dtype=np.float32)
        for key in ("observation.state.object", "observation.state.object_mask", "observation.state.placed_mask", "observation.state.task_progress"):
            obs_i[key] = np.asarray(default_state[key], dtype=np.float32)
        out_batch.append(obs_i)
    return out_batch


def _normalize_info_batched(info: dict[str, Any], num_envs: int) -> list[dict[str, Any]]:
    out = [dict() for _ in range(num_envs)]
    for k, v in info.items():
        arr = _to_numpy(v)
        if arr.shape == ():
            for i in range(num_envs):
                out[i][k] = float(arr)
            continue
        if arr.shape[0] != num_envs:
            for i in range(num_envs):
                out[i][k] = arr.tolist()
            continue
        for i in range(num_envs):
            item = arr[i]
            out[i][k] = float(item) if np.isscalar(item) else item.tolist()
    return out


def _collect_vectorized_scripted_episodes(
    ep_cfg: RootConfig,
    *,
    image_keys: list[str],
    state_keys: list[str],
    num_envs: int,
    episodes_target: int,
    max_steps: int,
    only_success: bool,
) -> list[tuple[_EpisodeBuffer, dict[str, Any]]]:
    import mini_pi0.sim.maniskill3_custom_env  # noqa: F401

    env_kwargs = dict(ep_cfg.simulator.env_kwargs or {})
    task_id = str(ep_cfg.simulator.task or "MiniPi0MultiObjectTray-v1")
    env = gym.make(
        task_id,
        num_envs=num_envs,
        obs_mode=env_kwargs.pop("obs_mode", "rgbd"),
        reward_mode=env_kwargs.pop("reward_mode", "dense"),
        control_mode=env_kwargs.pop("control_mode", str(ep_cfg.simulator.controller)),
        render_mode="none",
        render_backend=env_kwargs.pop("render_backend", "gpu"),
        sim_backend=env_kwargs.pop("sim_backend", "auto"),
        robot_uids=env_kwargs.pop("robot_uids", str(ep_cfg.simulator.robot).lower()),
        **env_kwargs,
    )
    raw_obs, _ = env.reset(seed=int(ep_cfg.experiment.seed))
    tray_center = np.asarray(ep_cfg.simulator.env_kwargs.get("tray_center", [0.62, 0.0, 0.0]), dtype=np.float32)
    policies = [ScriptedMultiObjectOracle(tray_center=tray_center) for _ in range(num_envs)]
    for p in policies:
        p.reset()
    buffers = [_EpisodeBuffer(obs=[], actions=[], rewards=[], dones=[], info_rows=[]) for _ in range(num_envs)]
    finalized: list[tuple[_EpisodeBuffer, dict[str, Any]]] = []

    for _ in range(int(max_steps)):
        obs_batch = _canonical_obs_batch_from_raw_env(env, image_keys=image_keys, state_keys=state_keys, raw_obs=raw_obs)
        acts = []
        for i in range(num_envs):
            obs_i = obs_batch[i]
            act_i = policies[i].act(obs_i).astype(np.float32)
            buffers[i].obs.append({
                k: np.asarray(obs_i[k])
                for k in set(image_keys + state_keys + [
                    "observation.state.object",
                    "observation.state.object_mask",
                    "observation.state.placed_mask",
                    "observation.state.task_progress",
                ])
                if k in obs_i
            })
            acts.append(act_i)
        actions_np = np.stack(acts, axis=0)
        raw_obs, reward, terminated, truncated, info = env.step(actions_np)
        rew = _to_numpy(reward).astype(np.float32).reshape(num_envs)
        ter = _to_numpy(terminated).astype(bool).reshape(num_envs)
        tru = _to_numpy(truncated).astype(bool).reshape(num_envs)
        info_rows = _normalize_info_batched(dict(info), num_envs=num_envs)

        for i in range(num_envs):
            buffers[i].actions.append(actions_np[i])
            buffers[i].rewards.append(float(rew[i]))
            done_i = bool(ter[i] or tru[i] or float(info_rows[i].get("success_fraction", 0.0)) >= 1.0 - 1e-6)
            buffers[i].dones.append(1 if done_i else 0)
            info_rows[i]["success"] = bool(float(info_rows[i].get("success_fraction", 0.0)) >= 1.0 - 1e-6)
            buffers[i].info_rows.append(info_rows[i])

            if done_i:
                fi = dict(info_rows[i])
                fi.setdefault("placed_count", int(np.sum(obs_batch[i]["observation.state.placed_mask"] > 0.5)))
                fi.setdefault("total_objects", int(np.sum(obs_batch[i]["observation.state.object_mask"] > 0.5)))
                if (not only_success) or bool(fi.get("success", False)):
                    finalized.append((buffers[i], fi))
                buffers[i] = _EpisodeBuffer(obs=[], actions=[], rewards=[], dones=[], info_rows=[])
                policies[i].reset()
                if len(finalized) >= int(episodes_target):
                    env.close()
                    return finalized
    if len(finalized) < int(episodes_target):
        for i in range(num_envs):
            if len(finalized) >= int(episodes_target):
                break
            if len(buffers[i].actions) == 0:
                continue
            obs_i = _canonical_obs_batch_from_raw_env(env, image_keys=image_keys, state_keys=state_keys, raw_obs=raw_obs)[i]
            fi = {
                "success": bool(float(obs_i["observation.state.task_progress"][0]) >= 1.0 - 1e-6),
                "success_fraction": float(obs_i["observation.state.task_progress"][0]),
                "placed_count": int(np.sum(obs_i["observation.state.placed_mask"] > 0.5)),
                "total_objects": int(np.sum(obs_i["observation.state.object_mask"] > 0.5)),
            }
            if (not only_success) or fi["success"]:
                buffers[i].dones[-1] = 1
                finalized.append((buffers[i], fi))
    env.close()
    return finalized


def _collect_vectorized_mplib_episodes(
    ep_cfg: RootConfig,
    *,
    image_keys: list[str],
    state_keys: list[str],
    num_envs: int,
    episodes_target: int,
    max_steps: int,
    only_success: bool,
) -> list[tuple[_EpisodeBuffer, dict[str, Any]]]:
    import mini_pi0.sim.maniskill3_custom_env  # noqa: F401
    import mplib

    env_kwargs = dict(ep_cfg.simulator.env_kwargs or {})
    env_kwargs["control_mode"] = "pd_joint_pos"
    env = gym.make(
        str(ep_cfg.simulator.task or "MiniPi0MultiObjectTray-v1"),
        num_envs=num_envs,
        obs_mode=env_kwargs.pop("obs_mode", "rgbd"),
        reward_mode=env_kwargs.pop("reward_mode", "dense"),
        control_mode=env_kwargs.pop("control_mode", "pd_joint_pos"),
        render_mode="none",
        render_backend=env_kwargs.pop("render_backend", "gpu"),
        sim_backend=env_kwargs.pop("sim_backend", "auto"),
        robot_uids=env_kwargs.pop("robot_uids", "panda"),
        **env_kwargs,
    )
    raw_obs, _ = env.reset(seed=int(ep_cfg.experiment.seed))
    uw = env.unwrapped
    lo = np.asarray(env.action_space.low, dtype=np.float32)
    hi = np.asarray(env.action_space.high, dtype=np.float32)
    if lo.ndim == 2:
        lo_vec = lo[0]
        hi_vec = hi[0]
    else:
        lo_vec = lo
        hi_vec = hi
    action_dim = int(lo_vec.shape[0])
    gripper_open = float(hi_vec[-1])
    gripper_closed = float(lo_vec[-1])

    urdf = ".venv/lib/python3.11/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf"
    srdf = ".venv/lib/python3.11/site-packages/mani_skill/assets/robots/panda/panda_v2.srdf"
    link_names = [lnk.get_name() for lnk in uw.agent.robot.get_links()]
    joint_names = [j.get_name() for j in uw.agent.robot.get_active_joints()]
    planners = []
    base_p = _to_numpy(uw.agent.robot.pose.p).astype(np.float64)
    base_q = _to_numpy(uw.agent.robot.pose.q).astype(np.float64)
    for i in range(num_envs):
        pl = mplib.Planner(urdf=urdf, srdf=srdf, user_link_names=link_names, user_joint_names=joint_names, move_group="panda_hand_tcp")
        pl.set_base_pose(mplib.Pose(base_p[i], base_q[i]))
        planners.append(pl)

    tray_center = np.asarray(ep_cfg.simulator.env_kwargs.get("tray_center", [-0.05, 0.20, 0.006]), dtype=np.float32)
    scripted = [ScriptedMultiObjectOracle(tray_center=tray_center) for _ in range(num_envs)]
    for p in scripted:
        p.reset()
    buffers = [_EpisodeBuffer(obs=[], actions=[], rewards=[], dones=[], info_rows=[]) for _ in range(num_envs)]
    finalized: list[tuple[_EpisodeBuffer, dict[str, Any]]] = []
    states = []
    for _ in range(num_envs):
        states.append(dict(mode="mplib", phase="select", target_idx=None, queue=[], fail=0, hold=0, grip=gripper_open))

    def _enqueue_plan(i: int, tgt_p: np.ndarray, tgt_q: np.ndarray, grip: float, planning_time: float = 0.2) -> bool:
        q_now = _to_numpy(uw.agent.robot.get_qpos())[i].astype(np.float64)
        goal = mplib.Pose(np.asarray(tgt_p, dtype=np.float64), np.asarray(tgt_q, dtype=np.float64))
        res = planners[i].plan_screw(goal, q_now, time_step=1 / 20)
        if res.get("status") != "Success":
            res = planners[i].plan_pose(goal, q_now, time_step=1 / 20, planning_time=planning_time)
        if res.get("status") != "Success":
            return False
        traj = np.asarray(res.get("position"), dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] == 0:
            return False
        for q7 in traj:
            act = np.zeros((action_dim,), dtype=np.float32)
            act[:7] = q7[:7]
            act[-1] = float(grip)
            states[i]["queue"].append(np.clip(act, lo_vec, hi_vec))
        states[i]["grip"] = float(grip)
        return True

    start_wall = time.time()
    for _ in range(int(max_steps)):
        if len(finalized) >= int(episodes_target):
            break
        if (time.time() - start_wall) > max(60.0, float(max_steps) * 0.4):
            break
        obs_batch = _canonical_obs_batch_from_raw_env(env, image_keys=image_keys, state_keys=state_keys, raw_obs=raw_obs)
        obj = np.stack([o["observation.state.object"] for o in obs_batch], axis=0).reshape(num_envs, -1, 3)
        obj_mask = np.stack([o["observation.state.object_mask"] for o in obs_batch], axis=0)
        placed_mask = np.stack([o["observation.state.placed_mask"] for o in obs_batch], axis=0)
        tcp_p = _to_numpy(uw.agent.tcp.pose.p).astype(np.float32)
        tcp_q = _to_numpy(uw.agent.tcp.pose.q).astype(np.float32)

        actions = []
        for i in range(num_envs):
            obs_i = obs_batch[i]
            buffers[i].obs.append({
                k: np.asarray(obs_i[k])
                for k in set(image_keys + state_keys + [
                    "observation.state.object",
                    "observation.state.object_mask",
                    "observation.state.placed_mask",
                    "observation.state.task_progress",
                ])
                if k in obs_i
            })
            st = states[i]
            act = None
            if st["queue"]:
                act = st["queue"].pop(0)
            elif st["mode"] == "scripted":
                act = scripted[i].act(obs_i).astype(np.float32)
            else:
                # mplib per-env state machine
                if st["phase"] == "select":
                    candidates = [j for j in range(obj.shape[1]) if obj_mask[i, j] > 0.5 and placed_mask[i, j] < 0.5]
                    if not candidates:
                        st["mode"] = "scripted"
                        act = scripted[i].act(obs_i).astype(np.float32)
                    else:
                        dists = [np.linalg.norm(obj[i, j] - tcp_p[i]) for j in candidates]
                        st["target_idx"] = int(candidates[int(np.argmin(dists))])
                        st["phase"] = "to_pregrasp"
                if act is None and st["mode"] == "mplib":
                    tid = st["target_idx"]
                    if tid is None:
                        st["phase"] = "select"
                    else:
                        tp = obj[i, tid].copy()
                        if st["phase"] == "to_pregrasp":
                            tgt = np.array([tp[0], tp[1], 0.20], dtype=np.float32)
                            ok = _enqueue_plan(i, tgt, tcp_q[i], gripper_open)
                            st["phase"] = "to_grasp" if ok else "select"
                            st["fail"] += 0 if ok else 1
                        elif st["phase"] == "to_grasp":
                            tgt = np.array([tp[0], tp[1], 0.065], dtype=np.float32)
                            ok = _enqueue_plan(i, tgt, tcp_q[i], gripper_open)
                            st["phase"] = "close" if ok else "select"
                            st["hold"] = 10
                            st["fail"] += 0 if ok else 1
                        elif st["phase"] == "close":
                            if st["hold"] > 0:
                                q_now = _to_numpy(uw.agent.robot.get_qpos())[i].astype(np.float32)
                                a = np.zeros((action_dim,), dtype=np.float32)
                                a[:7] = q_now[:7]
                                a[-1] = gripper_closed
                                act = np.clip(a, lo_vec, hi_vec)
                                st["hold"] -= 1
                            else:
                                st["phase"] = "to_lift"
                        elif st["phase"] == "to_lift":
                            tgt = np.array([tp[0], tp[1], 0.23], dtype=np.float32)
                            ok = _enqueue_plan(i, tgt, tcp_q[i], gripper_closed)
                            st["phase"] = "to_tray_above" if ok else "select"
                            st["fail"] += 0 if ok else 1
                        elif st["phase"] == "to_tray_above":
                            tgt = np.array([tray_center[0], tray_center[1], 0.23], dtype=np.float32)
                            ok = _enqueue_plan(i, tgt, tcp_q[i], gripper_closed)
                            st["phase"] = "to_tray_place" if ok else "select"
                            st["fail"] += 0 if ok else 1
                        elif st["phase"] == "to_tray_place":
                            tgt = np.array([tray_center[0], tray_center[1], 0.11], dtype=np.float32)
                            ok = _enqueue_plan(i, tgt, tcp_q[i], gripper_closed)
                            st["phase"] = "open" if ok else "select"
                            st["hold"] = 10
                            st["fail"] += 0 if ok else 1
                        elif st["phase"] == "open":
                            if st["hold"] > 0:
                                q_now = _to_numpy(uw.agent.robot.get_qpos())[i].astype(np.float32)
                                a = np.zeros((action_dim,), dtype=np.float32)
                                a[:7] = q_now[:7]
                                a[-1] = gripper_open
                                act = np.clip(a, lo_vec, hi_vec)
                                st["hold"] -= 1
                            else:
                                st["phase"] = "retreat"
                        elif st["phase"] == "retreat":
                            tgt = np.array([tray_center[0], tray_center[1], 0.25], dtype=np.float32)
                            ok = _enqueue_plan(i, tgt, tcp_q[i], gripper_open)
                            st["phase"] = "settle" if ok else "select"
                            st["hold"] = 6
                            st["fail"] += 0 if ok else 1
                        elif st["phase"] == "settle":
                            if st["hold"] > 0:
                                q_now = _to_numpy(uw.agent.robot.get_qpos())[i].astype(np.float32)
                                a = np.zeros((action_dim,), dtype=np.float32)
                                a[:7] = q_now[:7]
                                a[-1] = gripper_open
                                act = np.clip(a, lo_vec, hi_vec)
                                st["hold"] -= 1
                            else:
                                st["phase"] = "select"
                        if st["fail"] >= 6:
                            st["mode"] = "scripted"
                            scripted[i].reset()
                if act is None:
                    if st["queue"]:
                        act = st["queue"].pop(0)
                    elif st["mode"] == "scripted":
                        act = scripted[i].act(obs_i).astype(np.float32)
                    else:
                        q_now = _to_numpy(uw.agent.robot.get_qpos())[i].astype(np.float32)
                        a = np.zeros((action_dim,), dtype=np.float32)
                        a[:7] = q_now[:7]
                        a[-1] = st.get("grip", gripper_open)
                        act = np.clip(a, lo_vec, hi_vec)
            actions.append(act.astype(np.float32))

        actions_np = np.stack(actions, axis=0)
        raw_obs, reward, terminated, truncated, info = env.step(actions_np)
        rew = _to_numpy(reward).astype(np.float32).reshape(num_envs)
        ter = _to_numpy(terminated).astype(bool).reshape(num_envs)
        tru = _to_numpy(truncated).astype(bool).reshape(num_envs)
        info_rows = _normalize_info_batched(dict(info), num_envs=num_envs)
        for i in range(num_envs):
            buffers[i].actions.append(actions_np[i])
            buffers[i].rewards.append(float(rew[i]))
            done_i = bool(ter[i] or tru[i] or float(info_rows[i].get("success_fraction", 0.0)) >= 1.0 - 1e-6)
            buffers[i].dones.append(1 if done_i else 0)
            info_rows[i]["success"] = bool(float(info_rows[i].get("success_fraction", 0.0)) >= 1.0 - 1e-6)
            buffers[i].info_rows.append(info_rows[i])
            if done_i:
                fi = dict(info_rows[i])
                fi.setdefault("placed_count", int(np.sum(obs_batch[i]["observation.state.placed_mask"] > 0.5)))
                fi.setdefault("total_objects", int(np.sum(obs_batch[i]["observation.state.object_mask"] > 0.5)))
                if (not only_success) or bool(fi.get("success", False)):
                    finalized.append((buffers[i], fi))
                buffers[i] = _EpisodeBuffer(obs=[], actions=[], rewards=[], dones=[], info_rows=[])
                scripted[i].reset()
                states[i] = dict(mode="mplib", phase="select", target_idx=None, queue=[], fail=0, hold=0, grip=gripper_open)
                if len(finalized) >= int(episodes_target):
                    env.close()
                    return finalized
    env.close()
    return finalized


def _mplib_runtime_check() -> bool:
    code = (
        "import mplib\n"
        "urdf='"
        ".venv/lib/python3.11/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf"
        "'\n"
        "mplib.Planner(urdf=urdf, move_group='panda_hand_tcp')\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return proc.returncode == 0 and "ok" in proc.stdout


def _normalize_info(info: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in info.items():
        if hasattr(v, "detach"):
            arr = v.detach().cpu().numpy()
            if arr.shape == ():
                out[k] = float(arr)
            elif arr.shape[0] == 1:
                item = arr[0]
                out[k] = float(item) if np.isscalar(item) else item.tolist()
            else:
                out[k] = arr.tolist()
        else:
            out[k] = v
    return out


def _canonical_obs_from_raw_env(env, image_keys: list[str], state_keys: list[str], last_raw_obs: Any) -> dict[str, np.ndarray]:
    uw = env.unwrapped
    tcp_p = np.asarray(uw.agent.tcp.pose.p.detach().cpu().numpy())[0].astype(np.float32)
    tcp_q = np.asarray(uw.agent.tcp.pose.q.detach().cpu().numpy())[0].astype(np.float32)
    qpos = np.asarray(uw.agent.robot.get_qpos().detach().cpu().numpy())[0].astype(np.float32)

    obj_pos = np.asarray(uw._get_object_pos_tensor().detach().cpu().numpy())[0].reshape(-1).astype(np.float32)
    active = np.asarray(uw._active_object_mask.detach().cpu().numpy())[0].astype(np.float32)
    placed = np.asarray(uw._placed_mask.detach().cpu().numpy())[0].astype(np.float32)
    frac = float(uw._last_success_fraction[0].item()) if uw._last_success_fraction is not None else 0.0

    frame = None
    if isinstance(last_raw_obs, dict):
        try:
            rgb = last_raw_obs["sensor_data"]["base_camera"]["rgb"]
            arr = np.asarray(rgb)
            if arr.ndim == 4:
                arr = arr[0]
            frame = np.clip(arr, 0, 255).astype(np.uint8)
        except Exception:
            frame = None
    if frame is None:
        try:
            frame = np.asarray(env.render()).astype(np.uint8)
        except Exception:
            frame = np.zeros((128, 128, 3), dtype=np.uint8)

    default_state = {
        "robot0_eef_pos": tcp_p,
        "observation.state.eef_pos": tcp_p,
        "robot0_eef_quat": tcp_q,
        "observation.state.eef_quat": tcp_q,
        "robot0_gripper_qpos": qpos[-2:] if qpos.shape[0] >= 2 else qpos,
        "observation.state.tool": qpos[-2:] if qpos.shape[0] >= 2 else qpos,
        "observation.state.object": obj_pos,
        "observation.state.object_mask": active,
        "observation.state.placed_mask": placed,
        "observation.state.task_progress": np.array([frac], dtype=np.float32),
    }
    out: dict[str, np.ndarray] = {}
    for key in image_keys:
        out[key] = frame
    for key in state_keys:
        out[key] = np.asarray(default_state.get(key, np.zeros((1,), dtype=np.float32)), dtype=np.float32)
    for key in ("observation.state.object", "observation.state.object_mask", "observation.state.placed_mask", "observation.state.task_progress"):
        out[key] = np.asarray(default_state[key], dtype=np.float32)
    return out


def _collect_single_episode_mplib(ep_cfg: RootConfig, *, max_steps: int, image_keys: list[str], state_keys: list[str]):
    import mplib
    from mani_skill.examples.motionplanning.base_motionplanner.utils import (
        compute_grasp_info_by_obb,
        get_actor_obb,
    )

    env_kwargs = dict(ep_cfg.simulator.env_kwargs or {})
    env_kwargs["control_mode"] = "pd_joint_pos"
    env_kwargs["obs_mode"] = env_kwargs.get("obs_mode", "rgbd")
    env = gym.make(
        str(ep_cfg.simulator.task or "MiniPi0MultiObjectTray-v1"),
        obs_mode=env_kwargs.pop("obs_mode", "rgbd"),
        reward_mode=env_kwargs.pop("reward_mode", "dense"),
        control_mode=env_kwargs.pop("control_mode", "pd_joint_pos"),
        render_mode="rgb_array",
        render_backend=env_kwargs.pop("render_backend", "cpu"),
        sim_backend=env_kwargs.pop("sim_backend", "cpu"),
        robot_uids=env_kwargs.pop("robot_uids", "panda"),
        **env_kwargs,
    )
    raw_obs, _ = env.reset(seed=int(ep_cfg.experiment.seed))
    buf = _EpisodeBuffer(obs=[], actions=[], rewards=[], dones=[], info_rows=[])
    final_info: dict[str, Any] = {"success": False, "success_fraction": 0.0, "placed_count": 0, "total_objects": 0}
    step_counter = {"n": 0}

    orig_step = env.step
    def _recording_step(action):
        obs, reward, terminated, truncated, info = orig_step(action)
        step_counter["n"] += 1
        norm = _normalize_info(dict(info))
        can_obs = _canonical_obs_from_raw_env(env, image_keys=image_keys, state_keys=state_keys, last_raw_obs=obs)
        buf.obs.append(can_obs)
        buf.actions.append(np.asarray(action, dtype=np.float32))
        buf.rewards.append(float(np.asarray(reward).item()))
        done = bool(np.asarray(terminated).item() or np.asarray(truncated).item())
        if float(norm.get("success_fraction", 0.0)) >= 1.0 - 1e-6:
            done = True
        buf.dones.append(1 if done else 0)
        norm["success"] = bool(float(norm.get("success_fraction", 0.0)) >= 1.0 - 1e-6)
        buf.info_rows.append(norm)
        return obs, reward, terminated, truncated, info
    env.step = _recording_step

    uw = env.unwrapped
    tray_center = np.asarray(uw.tray_center_np, dtype=np.float32)
    urdf = ".venv/lib/python3.11/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf"
    srdf = ".venv/lib/python3.11/site-packages/mani_skill/assets/robots/panda/panda_v2.srdf"
    link_names = [lnk.get_name() for lnk in uw.agent.robot.get_links()]
    joint_names = [j.get_name() for j in uw.agent.robot.get_active_joints()]
    planner = mplib.Planner(
        urdf=urdf,
        srdf=srdf,
        user_link_names=link_names,
        user_joint_names=joint_names,
        move_group="panda_hand_tcp",
    )
    robot_base_p = uw.agent.robot.pose.p.detach().cpu().numpy()[0].astype(np.float64)
    robot_base_q = uw.agent.robot.pose.q.detach().cpu().numpy()[0].astype(np.float64)
    planner.set_base_pose(mplib.Pose(robot_base_p, robot_base_q))
    lo = np.asarray(env.action_space.low, dtype=np.float32)
    hi = np.asarray(env.action_space.high, dtype=np.float32)
    gripper_open = float(hi[-1])
    gripper_closed = float(lo[-1])

    def _exec_joint_path(result: dict[str, Any], gripper_cmd: float) -> None:
        if result.get("status") != "Success":
            return
        traj = np.asarray(result.get("position"), dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] == 0:
            return
        for q7 in traj:
            if step_counter["n"] >= int(max_steps):
                break
            action = np.zeros((env.action_space.shape[0],), dtype=np.float32)
            action[:7] = q7[:7]
            action[-1] = gripper_cmd
            env.step(np.clip(action, lo, hi))

    def _hold_gripper(gripper_cmd: float, n: int = 8) -> None:
        for _ in range(n):
            if step_counter["n"] >= int(max_steps):
                break
            q_now = uw.agent.robot.get_qpos().detach().cpu().numpy()[0].astype(np.float32)
            action = np.zeros((env.action_space.shape[0],), dtype=np.float32)
            action[:7] = q_now[:7]
            action[-1] = gripper_cmd
            env.step(np.clip(action, lo, hi))

    def _to_mplib_pose(sp_pose: sapien.Pose):
        return mplib.Pose(
            np.asarray(sp_pose.p, dtype=np.float64),
            np.asarray(sp_pose.q, dtype=np.float64),
        )

    def _move_pose(sp_pose: sapien.Pose, gripper_cmd: float) -> bool:
        q_now = uw.agent.robot.get_qpos().detach().cpu().numpy()[0].astype(np.float64)
        goal = _to_mplib_pose(sp_pose)
        res = planner.plan_screw(goal, q_now, time_step=1 / 20)
        if res.get("status") != "Success":
            res = planner.plan_pose(goal, q_now, time_step=1 / 20, planning_time=0.25)
        if res.get("status") != "Success":
            return False
        _exec_joint_path(res, gripper_cmd=gripper_cmd)
        return True

    start_wall = time.time()
    try:
        while step_counter["n"] < int(max_steps):
            if (time.time() - start_wall) > 45.0:
                break
            info = uw.evaluate()
            success_fraction = float(info["success_fraction"][0].item())
            if success_fraction >= 1.0 - 1e-6:
                break
            placed = np.asarray(info["placed_mask"].detach().cpu().numpy())[0]
            active = np.asarray(uw._active_object_mask.detach().cpu().numpy())[0]
            target_idx = None
            for i in range(min(len(active), len(placed))):
                if active[i] > 0 and placed[i] < 0.5:
                    target_idx = i
                    break
            if target_idx is None:
                break
            actor = uw.objects[int(target_idx)]
            actor_p = np.asarray(actor.pose.p.detach().cpu().numpy())[0]
            approaching = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            target_closing = (
                uw.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].detach().cpu().numpy()
            )
            obb = get_actor_obb(actor)
            grasp_info = compute_grasp_info_by_obb(
                obb,
                approaching=approaching,
                target_closing=target_closing,
                depth=0.025,
            )
            closing = grasp_info["closing"]
            center = grasp_info["center"]
            base_grasp_pose = uw.agent.build_grasp_pose(approaching, closing, center)

            # ManiSkill-style candidate search: rotate around z to find feasible approach.
            yaw_candidates = [0.0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2, 3 * np.pi / 4, -3 * np.pi / 4]
            grasp_pose = None
            pre_grasp = None
            for yaw in yaw_candidates:
                delta_q = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float64)
                cand = base_grasp_pose * sapien.Pose(q=delta_q)
                cand_pre = cand * sapien.Pose([0.0, 0.0, -0.07])
                q_now = uw.agent.robot.get_qpos().detach().cpu().numpy()[0].astype(np.float64)
                res_pre = planner.plan_screw(_to_mplib_pose(cand_pre), q_now, time_step=1 / 20)
                if res_pre.get("status") != "Success":
                    res_pre = planner.plan_pose(_to_mplib_pose(cand_pre), q_now, time_step=1 / 20, planning_time=0.2)
                if res_pre.get("status") == "Success":
                    grasp_pose = cand
                    pre_grasp = cand_pre
                    break
            if grasp_pose is None or pre_grasp is None:
                continue

            lift_pose = sapien.Pose([actor_p[0], actor_p[1], max(actor_p[2] + 0.14, 0.18)], grasp_pose.q)
            tray_above = sapien.Pose([tray_center[0], tray_center[1], 0.20], grasp_pose.q)
            tray_place = sapien.Pose([tray_center[0], tray_center[1], 0.095], grasp_pose.q)
            retreat = sapien.Pose([tray_center[0], tray_center[1], 0.22], grasp_pose.q)

            if not _move_pose(pre_grasp, gripper_open):
                continue
            if not _move_pose(grasp_pose, gripper_open):
                continue
            _hold_gripper(gripper_closed, n=10)
            if not _move_pose(lift_pose, gripper_closed):
                continue
            if not _move_pose(tray_above, gripper_closed):
                continue
            if not _move_pose(tray_place, gripper_closed):
                continue
            _hold_gripper(gripper_open, n=10)
            _move_pose(retreat, gripper_open)
            # Let placement settle so success mask can update.
            _hold_gripper(gripper_open, n=6)
            if step_counter["n"] >= int(max_steps):
                break
    finally:
        env.step = orig_step
        # gather final info snapshot
        fin = _normalize_info(dict(uw.evaluate()))
        final_info = {
            "success": bool(float(fin.get("success_fraction", 0.0)) >= 1.0 - 1e-6),
            "success_fraction": float(fin.get("success_fraction", 0.0)),
            "placed_count": int(float(fin.get("placed_count", 0.0))),
            "total_objects": int(float(fin.get("total_objects", 0.0))),
        }
        env.close()
    return buf, final_info
