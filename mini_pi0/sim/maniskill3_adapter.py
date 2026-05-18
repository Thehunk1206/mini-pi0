from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys
from mini_pi0.sim.base import SimulatorAdapter, StepOutput
from mini_pi0.sim.contact_features import collect_contact_features

# Register custom env on import.
import mini_pi0.sim.maniskill3_custom_env  # noqa: F401
import mini_pi0.sim.maniskill3_peginsertion_env  # noqa: F401


def default_maniskill_reward_mode(cfg: RootConfig) -> str:
    """Return a ManiSkill reward mode compatible with shaping settings.

    Args:
        cfg: Root runtime configuration.

    Returns:
        ``"dense"`` when shaped reward is requested, otherwise ``"sparse"``.
    """

    return "dense" if bool(getattr(cfg.simulator, "reward_shaping", True)) else "sparse"


def make_maniskill_env_with_reward_fallback(task_id: str, env_kwargs: dict[str, Any]):
    """Create a ManiSkill env, retrying sparse reward for unsupported dense tasks.

    Args:
        task_id: Gymnasium/ManiSkill task id.
        env_kwargs: Keyword arguments passed to ``gym.make``.

    Returns:
        Constructed Gymnasium environment.

    Raises:
        Exception: Re-raises non reward-mode construction errors.
    """

    try:
        return gym.make(task_id, **env_kwargs)
    except NotImplementedError as exc:
        reward_mode = str(env_kwargs.get("reward_mode", ""))
        if "Unsupported reward mode" not in str(exc) or reward_mode == "sparse":
            raise
        retry_kwargs = dict(env_kwargs)
        retry_kwargs["reward_mode"] = "sparse"
        print(
            f"[maniskill] reward_mode={reward_mode!r} is unsupported for {task_id}; retrying with 'sparse'.",
            flush=True,
        )
        return gym.make(task_id, **retry_kwargs)


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
        env_kwargs.pop("render_mode", None)
        render_backend = env_kwargs.pop("render_backend", "cpu")
        env_kwargs.pop("scripted_control_mode", None)

        control_mode = str(cfg.simulator.controller)
        if control_mode.strip().upper() == "BASIC":
            control_mode = "pd_ee_delta_pose"

        self.env = make_maniskill_env_with_reward_fallback(
            task_id,
            {
                "obs_mode": env_kwargs.pop("obs_mode", "state_dict"),
                "reward_mode": env_kwargs.pop("reward_mode", default_maniskill_reward_mode(cfg)),
                "control_mode": env_kwargs.pop("control_mode", control_mode),
                "render_mode": render_mode,
                "render_backend": render_backend,
                "sim_backend": env_kwargs.pop("sim_backend", "auto"),
                "robot_uids": env_kwargs.pop("robot_uids", str(cfg.simulator.robot).lower()),
                "max_episode_steps": env_kwargs.pop("max_episode_steps", int(cfg.simulator.horizon)),
                **env_kwargs,
            },
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
        if hasattr(uw, "_get_object_pos_tensor"):
            obj_pos = uw._get_object_pos_tensor()
            active = uw._active_object_mask.float()
            placed = uw._placed_mask.float()
            frac = float(uw._last_success_fraction[0].item()) if uw._last_success_fraction is not None else 0.0

            obj_pos_np = self._to_numpy(obj_pos)[0].reshape(-1)
            active_np = self._to_numpy(active)[0]
            placed_np = self._to_numpy(placed)[0]
            return obj_pos_np.astype(np.float32), active_np.astype(np.float32), placed_np.astype(np.float32), frac

        actor_names = [name for name in ("cubeA", "cubeB", "obj") if hasattr(uw, name)]
        actor_states = [self._actor_pose_vector(getattr(uw, name)) for name in actor_names]
        if actor_states:
            eval_info = self._evaluate_task()
            success = bool(eval_info.get("success", False))
            obj_state = np.concatenate(actor_states, axis=0).astype(np.float32)
            active = np.ones((len(actor_states),), dtype=np.float32)
            placed = np.ones((len(actor_states),), dtype=np.float32) if success else np.zeros((len(actor_states),), dtype=np.float32)
            return obj_state, active, placed, float(success)

        eval_info = self._evaluate_task()
        success = bool(eval_info.get("success", False))
        return (
            np.zeros((1,), dtype=np.float32),
            np.ones((1,), dtype=np.float32),
            np.ones((1,), dtype=np.float32) if success else np.zeros((1,), dtype=np.float32),
            float(success),
        )

    def _actor_pose_vector(self, actor: Any) -> np.ndarray:
        """Return one actor pose as a flat ``[x, y, z, qw, qx, qy, qz]`` vector."""

        pose = getattr(actor, "pose", None)
        raw_pose = getattr(pose, "raw_pose", None)
        if raw_pose is not None:
            arr = self._to_numpy(raw_pose)
        elif pose is not None and hasattr(pose, "p") and hasattr(pose, "q"):
            arr = np.concatenate([self._to_numpy(pose.p), self._to_numpy(pose.q)], axis=-1)
        else:
            return np.zeros((7,), dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim >= 2:
            arr = arr[0]
        return arr.reshape(-1)[:7].astype(np.float32)

    def _evaluate_task(self) -> dict[str, Any]:
        """Return scalarized ManiSkill task evaluation info when available."""

        uw = self.unwrapped
        if not hasattr(uw, "evaluate"):
            return {}
        try:
            raw = uw.evaluate()
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        return self._normalize_info(raw)

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
        qvel = self._to_numpy(uw.agent.robot.get_qvel())[0].astype(np.float32)

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
                cam_name = _camera_name_for_image_key(key)
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
            "robot0_joint_vel": qvel,
            "observation.state.joint_vel": qvel,
            "observation.state.object": obj_flat,
            "observation.state.object_mask": obj_mask,
            "observation.state.placed_mask": placed_mask,
            "observation.state.grasped_mask": grasped_mask,
            "observation.state.place_targets": place_targets_np,
            "observation.state.task_progress": np.array([frac], dtype=np.float32),
        }
        default_state.update(self._contact_state())

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

    def _contact_state(self) -> dict[str, np.ndarray]:
        """Return live compact contact features for configured proprio keys."""

        if not any(self._requires_contact_key(key) for key in self._state_keys):
            return {}
        try:
            return collect_contact_features(self.unwrapped)
        except Exception:
            return {}

    @staticmethod
    def _requires_contact_key(key: str) -> bool:
        """Return whether a state key is produced by contact feature extraction."""

        return (
            key.startswith("pair_")
            or key.endswith("_force")
            or key.endswith("_force_norm")
            or key.endswith("_contact")
            or key.endswith("_contact_count")
            or key.startswith("robot_qf")
            or key.startswith("robot_arm_qf")
            or key.startswith("robot_gripper_qf")
            or key.startswith("robot_passive_qf")
            or key.startswith("robot_arm_passive_qf")
            or key.startswith("robot_gripper_passive_qf")
        )

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
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        should_clip = bool(getattr(getattr(self.cfg, "eval", None), "action_clip", True))
        if should_clip:
            raw_action = np.clip(raw_action, lo, hi).astype(np.float32)
        raw_obs, reward, terminated, truncated, info = self.env.step(raw_action)
        self._last_raw_obs = raw_obs
        norm_info = self._normalize_info(dict(info))

        obs = self._canonical_obs_from_env()
        native_success = bool(norm_info.get("success", False))
        if "success_fraction" in norm_info:
            frac = float(norm_info["success_fraction"])
        else:
            frac = 1.0 if native_success else float(obs["observation.state.task_progress"][0])
        norm_info["success_fraction"] = frac
        norm_info["success"] = bool(native_success or frac >= 1.0 - 1e-6)

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
        frame = self._render_from_latest_sensor(camera=camera)
        if frame is not None:
            return _resize_uint8_frame(frame, width=width, height=height)
        try:
            frame = self.env.render()
            if isinstance(frame, np.ndarray):
                return _resize_uint8_frame(frame.astype(np.uint8), width=width, height=height)
        except Exception:
            pass
        return np.zeros((int(height), int(width), 3), dtype=np.uint8)

    def _render_from_latest_sensor(self, camera: str) -> np.ndarray | None:
        """Return the latest RGB sensor frame for a requested render camera."""

        sensor_frames = self._extract_sensor_frames()
        if not sensor_frames:
            return None
        candidates = [str(camera)]
        mapped = _camera_name_for_image_key(str(camera))
        if mapped != str(camera):
            candidates.append(mapped)
        if not str(camera).endswith("_camera"):
            candidates.append(f"{camera}_camera")
        for name in self._preferred_camera_names:
            candidates.append(name)
        candidates.extend(sensor_frames.keys())
        for name in candidates:
            if name in sensor_frames:
                return np.asarray(sensor_frames[name], dtype=np.uint8)
        return None

    def check_success(self, info: dict[str, Any] | None = None, obs: dict[str, np.ndarray] | None = None) -> bool:
        src = info if info is not None else self._last_info
        if bool(src.get("success", False)):
            return True
        frac = float(src.get("success_fraction", 0.0))
        return bool(frac >= 1.0 - 1e-6)

    def close(self) -> None:
        self.env.close()


def _resize_uint8_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an RGB frame to the requested output size."""

    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        return np.zeros((int(height), int(width), 3), dtype=np.uint8)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    target_h = int(height)
    target_w = int(width)
    if arr.shape[0] == target_h and arr.shape[1] == target_w:
        return arr
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()


def _camera_name_for_image_key(image_key: str) -> str:
    """Map canonical mini-pi0 image keys to ManiSkill camera names."""

    aliases = {
        "agentview_image": "base_camera",
        "base_image": "base_camera",
        "robot0_eye_in_hand_image": "hand_camera",
        "hand_image": "hand_camera",
    }
    if image_key in aliases:
        return aliases[image_key]
    if image_key.endswith("_image"):
        return f"{image_key[:-6]}_camera"
    return image_key
