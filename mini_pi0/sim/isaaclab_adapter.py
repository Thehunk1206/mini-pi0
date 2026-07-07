from __future__ import annotations

"""Isaac Lab simulator adapter.

This module intentionally keeps Isaac imports lazy. Importing mini-pi0 on a
machine without Isaac Sim/Lab must remain safe; the adapter only requires Isaac
when it is constructed or smoke-tested.
"""

import argparse
import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys
from mini_pi0.sim.base import SimulatorAdapter, StepOutput
from mini_pi0.sim.isaaclab_tasks import IsaacLabTaskSpec, resolve_isaaclab_task


class IsaacLabUnavailableError(RuntimeError):
    """Raised when Isaac Lab code is requested outside an Isaac runtime."""


@dataclass(frozen=True)
class IsaacLabRuntime:
    """Lazy-loaded Isaac Lab runtime modules."""

    gym: Any
    task_spec: IsaacLabTaskSpec


_APP_LAUNCHER: Any | None = None


def isaaclab_available() -> bool:
    """Return whether the Python environment can import Isaac Lab."""

    return importlib.util.find_spec("isaaclab") is not None


def _load_runtime(task_name: str, *, headless: bool, enable_cameras: bool) -> IsaacLabRuntime:
    """Import Isaac Lab and launch its app when needed.

    Args:
        task_name: mini-pi0 task key, alias, or direct Isaac Lab Gym id.
        headless: Run Isaac Sim without an interactive window.
        enable_cameras: Enable camera sensors for image observations.

    Returns:
        Runtime module bundle.

    Raises:
        IsaacLabUnavailableError: If Isaac Lab is not importable.
    """

    if not isaaclab_available():
        raise IsaacLabUnavailableError(
            "Isaac Lab is not installed in this Python environment. "
            "Run this command inside the mini-pi0 Isaac Lab Docker container."
        )

    _launch_app(headless=headless, enable_cameras=enable_cameras)

    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401 - registers Isaac Lab Gym tasks.
    except Exception as exc:
        raise IsaacLabUnavailableError(
            "Failed to import Isaac Lab task registration modules inside the current runtime."
        ) from exc
    return IsaacLabRuntime(gym=gym, task_spec=resolve_isaaclab_task(task_name))


def _launch_app(*, headless: bool, enable_cameras: bool) -> None:
    """Launch the Isaac Lab app once per process."""

    global _APP_LAUNCHER
    if _APP_LAUNCHER is not None:
        return
    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:
        raise IsaacLabUnavailableError("Isaac Lab AppLauncher is unavailable.") from exc

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    argv = ["--headless"] if headless else []
    if enable_cameras:
        argv.append("--enable_cameras")
    args, _unknown = parser.parse_known_args(argv)
    _APP_LAUNCHER = AppLauncher(args)


class IsaacLabAdapter(SimulatorAdapter):
    """Isaac Lab backend adapter using the common mini-pi0 simulator API."""

    backend_name = "isaaclab"

    def __init__(self, cfg: RootConfig):
        """Create an Isaac Lab environment from config.

        Args:
            cfg: Root runtime configuration.

        Raises:
            IsaacLabUnavailableError: If Isaac Lab is not available.
        """

        self.cfg = cfg
        self._state_keys = effective_state_keys(cfg.robot)
        self._image_keys = effective_image_keys(cfg.robot)
        self._last_raw_obs: Any = None
        self._last_info: dict[str, Any] = {}
        self._last_obs: dict[str, np.ndarray] | None = None

        env_kwargs = dict(cfg.simulator.env_kwargs or {})
        headless = bool(env_kwargs.pop("headless", not bool(cfg.simulator.has_renderer)))
        enable_cameras = bool(env_kwargs.pop("enable_cameras", cfg.simulator.use_camera_obs))
        runtime = _load_runtime(cfg.simulator.task, headless=headless, enable_cameras=enable_cameras)
        self.task_spec = runtime.task_spec

        self.env = runtime.gym.make(
            self.task_spec.gym_id,
            render_mode=env_kwargs.pop("render_mode", "rgb_array" if cfg.simulator.has_offscreen_renderer else None),
            **env_kwargs,
        )

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset the Isaac Lab environment and return canonical observation."""

        raw = self.env.reset(seed=seed)
        raw_obs, info = raw if isinstance(raw, tuple) and len(raw) == 2 else (raw, {})
        self._last_raw_obs = raw_obs
        self._last_info = self._normalize_info(info if isinstance(info, dict) else {})
        obs = self._canonical_obs(raw_obs)
        self._last_obs = obs
        return obs

    def step(self, action: np.ndarray) -> StepOutput:
        """Step the Isaac Lab environment with one raw mini-pi0 action."""

        env_action = self._format_action(action)
        raw = self.env.step(env_action)
        if not isinstance(raw, tuple) or len(raw) != 5:
            raise RuntimeError("Isaac Lab Gym environment returned an unsupported step tuple.")
        raw_obs, reward, terminated, truncated, info = raw
        self._last_raw_obs = raw_obs
        norm_info = self._normalize_info(info if isinstance(info, dict) else {})
        obs = self._canonical_obs(raw_obs)
        done = bool(_scalarize(terminated) or _scalarize(truncated))
        norm_info.setdefault("success", self._success_from_info_or_obs(norm_info, obs))
        self._last_info = norm_info
        self._last_obs = obs
        return StepOutput(obs=obs, reward=float(_scalarize(reward)), done=done, info=norm_info)

    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        """Return flattened action bounds for one environment."""

        space = getattr(self.env, "single_action_space", None) or getattr(self.env, "action_space", None)
        if space is None:
            dim = int(max(1, self.cfg.robot.action_dim))
            return -np.ones((dim,), dtype=np.float32), np.ones((dim,), dtype=np.float32)
        low = np.asarray(getattr(space, "low", -1.0), dtype=np.float32).reshape(-1)
        high = np.asarray(getattr(space, "high", 1.0), dtype=np.float32).reshape(-1)
        if low.size != high.size:
            dim = int(max(low.size, high.size, self.cfg.robot.action_dim))
            low = np.resize(low, dim)
            high = np.resize(high, dim)
        return low.astype(np.float32), high.astype(np.float32)

    def render(self, camera: str = "agentview", width: int = 512, height: int = 512) -> np.ndarray:
        """Render the latest Isaac Lab RGB frame."""

        latest = self._image_from_raw(self._last_raw_obs)
        if latest is not None:
            return _resize_nearest(latest, width=width, height=height)
        render = getattr(self.env, "render", None)
        if callable(render):
            frame = render()
            if isinstance(frame, np.ndarray):
                return _resize_nearest(_as_uint8_rgb(frame), width=width, height=height)
        return np.zeros((int(height), int(width), 3), dtype=np.uint8)

    def check_success(self, info: dict[str, Any] | None = None, obs: dict[str, np.ndarray] | None = None) -> bool:
        """Return whether the current Isaac Lab task is solved."""

        src = info if info is not None else self._last_info
        return self._success_from_info_or_obs(src, obs if obs is not None else self._last_obs)

    def close(self) -> None:
        """Release Isaac Lab environment resources."""

        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _format_action(self, action: np.ndarray) -> np.ndarray:
        """Clip and shape one action for the Isaac Lab Gym environment."""

        low, high = self.action_spec()
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < low.size:
            padded = np.zeros_like(low, dtype=np.float32)
            padded[: arr.size] = arr
            arr = padded
        elif arr.size > low.size:
            arr = arr[: low.size]
        arr = np.clip(arr, low, high).astype(np.float32)
        shape = getattr(getattr(self.env, "action_space", None), "shape", None)
        if isinstance(shape, tuple) and len(shape) > 1 and shape[0] != arr.size:
            return arr.reshape((1, -1))
        return arr

    def _canonical_obs(self, raw_obs: Any) -> dict[str, np.ndarray]:
        """Map Isaac observations into mini-pi0 canonical observation keys."""

        frame = self._image_from_raw(raw_obs)
        if frame is None:
            frame = np.zeros(
                (int(self.cfg.simulator.camera_height), int(self.cfg.simulator.camera_width), 3),
                dtype=np.uint8,
            )
        state_source = _flatten_named_arrays(raw_obs)
        out: dict[str, np.ndarray] = {key: frame for key in self._image_keys}
        aliases = self._state_aliases(state_source)
        for key in self._state_keys:
            out[key] = np.asarray(aliases.get(key, np.zeros((1,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        for key in (
            "observation.state.object",
            "observation.state.task_progress",
            "observation.state.success",
        ):
            if key in aliases:
                out[key] = np.asarray(aliases[key], dtype=np.float32).reshape(-1)
        return out

    def _state_aliases(self, state_source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Build canonical state aliases from available Isaac observation arrays."""

        policy = _first_available(state_source, ("policy", "obs", "observations"))
        joint_pos = _first_available(state_source, ("joint_pos", "robot_joint_pos", "qpos"))
        joint_vel = _first_available(state_source, ("joint_vel", "robot_joint_vel", "qvel"))
        eef_pos = _first_available(state_source, ("eef_pos", "ee_pos", "end_effector_pos", "tcp_pos"))
        eef_quat = _first_available(state_source, ("eef_quat", "ee_quat", "end_effector_quat", "tcp_quat"))
        gripper = _first_available(state_source, ("gripper_pos", "gripper_qpos", "finger_pos"))
        obj = _first_available(state_source, ("object_pos", "cube_pos", "target_pos", "object"))
        progress = _first_available(state_source, ("success", "task_progress", "is_success"))

        aliases: dict[str, np.ndarray] = {}
        aliases["observation.state.policy"] = policy if policy is not None else np.zeros((1,), dtype=np.float32)
        aliases["robot0_joint_pos"] = joint_pos if joint_pos is not None else np.zeros((1,), dtype=np.float32)
        aliases["robot0_joint_vel"] = joint_vel if joint_vel is not None else np.zeros((1,), dtype=np.float32)
        aliases["robot0_eef_pos"] = eef_pos if eef_pos is not None else _slice_or_zero(policy, 0, 3)
        aliases["robot0_eef_quat"] = eef_quat if eef_quat is not None else _quat_identity()
        aliases["robot0_gripper_qpos"] = gripper if gripper is not None else _slice_or_zero(policy, 7, 9)
        aliases["observation.state.object"] = obj if obj is not None else _slice_or_zero(policy, 9, 12)
        aliases["observation.state.task_progress"] = progress if progress is not None else np.zeros((1,), dtype=np.float32)
        aliases["observation.state.success"] = aliases["observation.state.task_progress"]
        aliases.update(state_source)
        return aliases

    def _image_from_raw(self, raw_obs: Any) -> np.ndarray | None:
        """Extract an RGB frame from nested Isaac observations."""

        for key in ("rgb", "image", "camera", "front", "agentview_image"):
            value = _find_nested(raw_obs, key)
            if value is None:
                continue
            arr = np.asarray(_to_numpy(value))
            if arr.ndim >= 3:
                return _as_uint8_rgb(arr)
        return None

    def _success_from_info_or_obs(
        self,
        info: dict[str, Any] | None,
        obs: dict[str, np.ndarray] | None,
    ) -> bool:
        """Infer success from common Isaac info or canonical observation keys."""

        src = info or {}
        for key in ("success", "is_success", "terminated_success"):
            if key in src:
                return bool(_scalarize(src[key]))
        if obs is not None:
            for key in ("observation.state.success", "observation.state.task_progress"):
                if key in obs:
                    return bool(float(np.asarray(obs[key]).reshape(-1)[0]) >= 1.0)
        return False

    def _normalize_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """Convert tensors/arrays in Isaac info dictionaries into JSON-friendly values."""

        out: dict[str, Any] = {}
        for key, value in info.items():
            arr = _to_numpy(value)
            if isinstance(arr, np.ndarray):
                if arr.shape == ():
                    out[str(key)] = float(arr)
                elif arr.size == 1:
                    out[str(key)] = float(arr.reshape(-1)[0])
                else:
                    out[str(key)] = arr.tolist()
            else:
                out[str(key)] = arr
        return out


def _to_numpy(value: Any) -> Any:
    """Best-effort conversion of tensors to numpy without importing torch."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        return numpy()
    return value


def _scalarize(value: Any) -> float:
    """Return a scalar float from a tensor/array-like value."""

    arr = np.asarray(_to_numpy(value))
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])


def _find_nested(value: Any, target_key: str) -> Any | None:
    """Find a key in nested mappings/lists."""

    if isinstance(value, dict):
        if target_key in value:
            return value[target_key]
        for child in value.values():
            found = _find_nested(child, target_key)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for child in value:
            found = _find_nested(child, target_key)
            if found is not None:
                return found
    return None


def _flatten_named_arrays(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten nested mapping arrays into a name-to-array dictionary."""

    out: dict[str, np.ndarray] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_named_arrays(child, name))
        return out
    arr = np.asarray(_to_numpy(value), dtype=np.float32)
    if arr.size == 0:
        return out
    if arr.ndim >= 2:
        arr = arr.reshape(arr.shape[0], -1)[0] if arr.shape[0] == 1 else arr.reshape(-1)
    out[prefix] = arr.reshape(-1).astype(np.float32)
    leaf = prefix.split(".")[-1]
    out.setdefault(leaf, out[prefix])
    return out


def _first_available(source: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    """Return the first available array for a set of alias names."""

    for name in names:
        if name in source:
            return np.asarray(source[name], dtype=np.float32).reshape(-1)
    return None


def _slice_or_zero(value: np.ndarray | None, start: int, stop: int) -> np.ndarray:
    """Slice an array or return zeros when unavailable."""

    width = max(1, int(stop - start))
    if value is None:
        return np.zeros((width,), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size >= stop:
        return arr[start:stop]
    return np.zeros((width,), dtype=np.float32)


def _quat_identity() -> np.ndarray:
    """Return identity quaternion in wxyz convention."""

    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _as_uint8_rgb(value: np.ndarray) -> np.ndarray:
    """Normalize an array into ``[H, W, 3]`` uint8 RGB."""

    arr = np.asarray(value)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _resize_nearest(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Resize RGB frame with nearest-neighbor numpy indexing."""

    arr = _as_uint8_rgb(frame)
    target_h = int(max(1, height))
    target_w = int(max(1, width))
    if arr.shape[:2] == (target_h, target_w):
        return arr
    y = np.linspace(0, arr.shape[0] - 1, target_h).astype(np.int64)
    x = np.linspace(0, arr.shape[1] - 1, target_w).astype(np.int64)
    return arr[y][:, x]
