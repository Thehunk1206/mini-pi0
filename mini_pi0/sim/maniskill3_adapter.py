from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys
from mini_pi0.sim.base import SimulatorAdapter, StepOutput

# Register custom env on import.
import mini_pi0.sim.maniskill3_custom_env  # noqa: F401


class ManiSkill3Adapter(SimulatorAdapter):
    """ManiSkill3 backend adapter using custom registered task env."""

    backend_name = "maniskill3"

    def __init__(self, cfg: RootConfig):
        self.cfg = cfg
        self._state_keys = effective_state_keys(cfg.robot)
        self._image_keys = effective_image_keys(cfg.robot)
        self._preferred_camera_names = [
            str(name).strip()
            for name in (cfg.simulator.camera_names or [])
            if str(name).strip()
        ]

        env_kwargs = dict(cfg.simulator.env_kwargs or {})
        task_id = str(cfg.simulator.task or "MiniPi0MultiObjectTray-v1")
        if task_id.strip().lower() in {"pickcube-v1", "pickcube", "custom", "mini_pi0_multiobject", "lift"}:
            task_id = "MiniPi0MultiObjectTray-v1"
        try:
            gym.spec(task_id)
        except Exception:
            task_id = "MiniPi0MultiObjectTray-v1"

        render_mode = "rgb_array" if bool(cfg.simulator.has_offscreen_renderer) else "none"
        render_backend = env_kwargs.pop("render_backend", "cpu")
        env_kwargs.pop("scripted_control_mode", None)

        self.env = gym.make(
            task_id,
            obs_mode=env_kwargs.pop("obs_mode", "state_dict"),
            reward_mode=env_kwargs.pop("reward_mode", "dense"),
            control_mode=env_kwargs.pop("control_mode", str(cfg.simulator.controller)),
            render_mode=render_mode,
            render_backend=render_backend,
            sim_backend=env_kwargs.pop("sim_backend", "auto"),
            robot_uids=env_kwargs.pop("robot_uids", str(cfg.simulator.robot).lower()),
            **env_kwargs,
        )
        self._last_info: dict[str, Any] = {}
        self._last_obs: dict[str, np.ndarray] | None = None
        self._last_raw_obs: Any = None

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def _to_numpy(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _get_object_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        uw = self.unwrapped
        obj_pos = uw._get_object_pos_tensor()
        active = uw._active_object_mask.float()
        placed = uw._placed_mask.float()
        frac = float(uw._last_success_fraction[0].item()) if uw._last_success_fraction is not None else 0.0

        obj_pos_np = self._to_numpy(obj_pos)[0].reshape(-1)
        active_np = self._to_numpy(active)[0]
        placed_np = self._to_numpy(placed)[0]
        return obj_pos_np.astype(np.float32), active_np.astype(np.float32), placed_np.astype(np.float32), frac

    def _extract_sensor_frames(self) -> dict[str, np.ndarray]:
        raw = self._last_raw_obs
        if not isinstance(raw, dict):
            return {}
        out: dict[str, np.ndarray] = {}
        try:
            sensor_data = raw.get("sensor_data", {})
            if not isinstance(sensor_data, dict):
                return {}
            for camera_name, camera_payload in sensor_data.items():
                if not isinstance(camera_payload, dict):
                    continue
                rgb = camera_payload.get("rgb", None)
                if rgb is None:
                    continue
                arr = self._to_numpy(rgb)
                if arr.ndim == 5:
                    arr = arr[0, 0]
                elif arr.ndim == 4:
                    arr = arr[0]
                if arr.ndim != 3:
                    continue
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                out[str(camera_name)] = arr
            return out
        except Exception:
            return {}

    def _canonical_obs_from_env(self) -> dict[str, np.ndarray]:
        uw = self.unwrapped
        tcp_p = self._to_numpy(uw.agent.tcp.pose.p)[0].astype(np.float32)
        tcp_q = self._to_numpy(uw.agent.tcp.pose.q)[0].astype(np.float32)
        qpos = self._to_numpy(uw.agent.robot.get_qpos())[0].astype(np.float32)

        obj_flat, obj_mask, placed_mask, frac = self._get_object_state()
        grasped_mask = np.zeros_like(placed_mask, dtype=np.float32)
        try:
            grasp_rows = [self._to_numpy(uw.agent.is_grasping(actor))[0] for actor in uw.objects]
            if grasp_rows:
                grasped_mask = np.asarray(grasp_rows, dtype=np.float32)
        except Exception:
            grasped_mask = np.zeros_like(placed_mask, dtype=np.float32)

        sensor_frames = self._extract_sensor_frames()
        place_targets = getattr(uw, "_placement_targets", None)
        if place_targets is not None:
            place_targets_np = self._to_numpy(place_targets)[0].reshape(-1).astype(np.float32)
        else:
            place_targets_np = np.zeros_like(obj_flat, dtype=np.float32)
        frame = None
        camera_order: list[str] = []
        for name in self._preferred_camera_names:
            if name in sensor_frames and name not in camera_order:
                camera_order.append(name)
        for name in sensor_frames.keys():
            if name not in camera_order:
                camera_order.append(name)
        if camera_order:
            frame = sensor_frames[camera_order[0]]
        if bool(self.cfg.simulator.has_offscreen_renderer):
            if frame is None:
                try:
                    frame = self.env.render()
                except Exception:
                    frame = None
        if frame is None or not isinstance(frame, np.ndarray):
            frame = np.zeros((int(self.cfg.simulator.camera_height), int(self.cfg.simulator.camera_width), 3), dtype=np.uint8)

        out: dict[str, np.ndarray] = {}
        if not camera_order:
            for key in self._image_keys:
                out[key] = np.asarray(frame, dtype=np.uint8)
        else:
            for idx, key in enumerate(self._image_keys):
                cam_name = f"{key[:-6]}_camera" if key.endswith("_image") else camera_order[min(idx, len(camera_order) - 1)]
                if cam_name not in sensor_frames:
                    cam_name = camera_order[min(idx, len(camera_order) - 1)]
                out[key] = np.asarray(sensor_frames.get(cam_name, frame), dtype=np.uint8)

        default_state = {
            "robot0_eef_pos": tcp_p,
            "observation.state.eef_pos": tcp_p,
            "robot0_eef_quat": tcp_q,
            "observation.state.eef_quat": tcp_q,
            "robot0_gripper_qpos": qpos[-2:] if qpos.shape[0] >= 2 else qpos,
            "observation.state.tool": qpos[-2:] if qpos.shape[0] >= 2 else qpos,
            "observation.state.object": obj_flat,
            "observation.state.object_mask": obj_mask,
            "observation.state.placed_mask": placed_mask,
            "observation.state.grasped_mask": grasped_mask,
            "observation.state.place_targets": place_targets_np,
            "observation.state.task_progress": np.array([frac], dtype=np.float32),
        }

        for key in self._state_keys:
            out[key] = np.asarray(default_state.get(key, np.zeros((1,), dtype=np.float32)), dtype=np.float32)

        for key in (
            "observation.state.object",
            "observation.state.object_mask",
            "observation.state.placed_mask",
            "observation.state.grasped_mask",
            "observation.state.place_targets",
            "observation.state.task_progress",
        ):
            out[key] = np.asarray(default_state[key], dtype=np.float32)

        return out

    def _normalize_info(self, info: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                arr = v.detach().cpu().numpy()
                if arr.shape == ():
                    out[k] = float(arr)
                elif arr.shape[0] == 1:
                    item = arr[0]
                    if np.isscalar(item):
                        out[k] = float(item)
                    else:
                        out[k] = item.tolist()
                else:
                    out[k] = arr.tolist()
            else:
                out[k] = v
        return out

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        raw_obs, _info = self.env.reset(seed=seed)
        self._last_raw_obs = raw_obs
        obs = self._canonical_obs_from_env()
        _, _, _, frac = self._get_object_state()
        obj_mask = obs["observation.state.object_mask"]
        total_objects = int(np.sum(obj_mask > 0.5))
        placed_count = int(np.sum(obs["observation.state.placed_mask"] > 0.5))
        self._last_info = {
            "success": bool(frac >= 1.0 - 1e-6),
            "success_fraction": float(frac),
            "placed_count": placed_count,
            "total_objects": total_objects,
        }
        self._last_obs = obs
        return obs

    def step(self, action: np.ndarray) -> StepOutput:
        lo, hi = self.action_spec()
        clipped = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), lo, hi)
        raw_obs, reward, terminated, truncated, info = self.env.step(clipped)
        self._last_raw_obs = raw_obs
        norm_info = self._normalize_info(dict(info))

        obs = self._canonical_obs_from_env()
        frac = float(norm_info.get("success_fraction", obs["observation.state.task_progress"][0]))
        norm_info["success_fraction"] = frac
        norm_info["success"] = bool(frac >= 1.0 - 1e-6)

        if "placed_count" not in norm_info:
            norm_info["placed_count"] = int(np.sum(obs["observation.state.placed_mask"] > 0.5))
        if "total_objects" not in norm_info:
            norm_info["total_objects"] = int(np.sum(obs["observation.state.object_mask"] > 0.5))

        self._last_info = norm_info
        self._last_obs = obs

        done = bool(np.asarray(terminated).item() or np.asarray(truncated).item())
        return StepOutput(obs=obs, reward=float(np.asarray(reward).item()), done=done, info=norm_info)

    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        space = self.env.action_space
        lo = np.asarray(space.low, dtype=np.float32).reshape(-1)
        hi = np.asarray(space.high, dtype=np.float32).reshape(-1)
        return lo, hi

    def render(self, camera: str = "agentview", width: int = 512, height: int = 512) -> np.ndarray:
        try:
            frame = self.env.render()
            if isinstance(frame, np.ndarray):
                return frame.astype(np.uint8)
        except Exception:
            pass
        return np.zeros((int(height), int(width), 3), dtype=np.uint8)

    def check_success(self, info: dict[str, Any] | None = None, obs: dict[str, np.ndarray] | None = None) -> bool:
        src = info if info is not None else self._last_info
        frac = float(src.get("success_fraction", 0.0))
        return bool(frac >= 1.0 - 1e-6)

    def close(self) -> None:
        self.env.close()
