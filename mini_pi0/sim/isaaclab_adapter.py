"""Isaac Lab simulator adapter.

This module intentionally keeps Isaac imports lazy. Importing mini-pi0 on a
machine without Isaac Sim/Lab must remain safe; the adapter only requires Isaac
when it is constructed or smoke-tested.
"""

from __future__ import annotations

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
    parse_env_cfg: Any
    task_spec: IsaacLabTaskSpec


_APP_LAUNCHER: Any | None = None
_FRONT_CAMERA_NAME = "mini_pi0_front_camera"


def isaaclab_available() -> bool:
    """Return whether the Python environment can import Isaac Lab."""

    return importlib.util.find_spec("isaaclab") is not None


def _configure_front_camera(env_cfg: Any, *, width: int, height: int) -> None:
    """Attach a fixed tiled RGB camera to an Isaac scene configuration."""

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/MiniPi0FrontCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.1, 0.0, 0.8),
            rot=(0.250081, -0.661407, -0.661407, 0.250081),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
        width=width,
        height=height,
    )
    setattr(env_cfg.scene, _FRONT_CAMERA_NAME, camera)


def _finite_action_bounds(space: Any, *, fallback_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return finite bounds, interpreting Isaac's unbounded Box as normalized commands."""

    dim = max(1, int(fallback_dim))
    if space is None:
        return -np.ones(dim, dtype=np.float32), np.ones(dim, dtype=np.float32)
    low = np.asarray(getattr(space, "low", -1.0), dtype=np.float32).reshape(-1)
    high = np.asarray(getattr(space, "high", 1.0), dtype=np.float32).reshape(-1)
    if low.size != high.size or low.size == 0:
        return -np.ones(dim, dtype=np.float32), np.ones(dim, dtype=np.float32)
    low = np.where(np.isfinite(low), low, -1.0).astype(np.float32)
    high = np.where(np.isfinite(high), high, 1.0).astype(np.float32)
    if not np.all(low < high):
        raise ValueError("Invalid Isaac Lab action bounds after normalization.")
    return low, high


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
        from isaaclab_tasks.utils import parse_env_cfg
    except Exception as exc:
        raise IsaacLabUnavailableError(
            "Failed to import Isaac Lab task registration modules inside the current runtime."
        ) from exc
    return IsaacLabRuntime(gym=gym, parse_env_cfg=parse_env_cfg, task_spec=resolve_isaaclab_task(task_name))


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

        env_cfg = runtime.parse_env_cfg(
            self.task_spec.gym_id,
            device=str(env_kwargs.pop("device", "cuda:0")),
            num_envs=int(env_kwargs.pop("num_envs", 1)),
            use_fabric=bool(env_kwargs.pop("use_fabric", True)),
        )
        if enable_cameras and cfg.simulator.use_camera_obs:
            _configure_front_camera(
                env_cfg,
                width=int(cfg.simulator.camera_width),
                height=int(cfg.simulator.camera_height),
            )
        render_mode = env_kwargs.pop("render_mode", "rgb_array" if cfg.simulator.has_offscreen_renderer else None)
        make_kwargs = {"cfg": env_cfg, **env_kwargs}
        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode
        self.env = runtime.gym.make(
            self.task_spec.gym_id,
            **make_kwargs,
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
        norm_info.setdefault("success", self._success_from_info_or_obs(norm_info, obs))
        self._last_info = norm_info
        self._last_obs = obs
        return StepOutput(
            obs=obs,
            reward=float(_scalarize(reward)),
            terminated=bool(_scalarize(terminated)),
            truncated=bool(_scalarize(truncated)),
            info=norm_info,
        )

    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        """Return flattened action bounds for one environment."""

        space = getattr(self.env, "single_action_space", None) or getattr(self.env, "action_space", None)
        return _finite_action_bounds(space, fallback_dim=int(self.cfg.robot.action_dim))

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

    def _format_action(self, action: np.ndarray) -> Any:
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
            arr = arr.reshape((1, -1))
        return self._to_env_action(arr)

    def _to_env_action(self, action: np.ndarray) -> Any:
        """Convert a clipped NumPy action to the tensor type expected by Isaac Lab."""

        try:
            import torch
        except ModuleNotFoundError:
            return action
        device = getattr(getattr(self.env, "unwrapped", self.env), "device", None)
        return torch.as_tensor(action, dtype=torch.float32, device=device)

    def _canonical_obs(self, raw_obs: Any, env_index: int = 0) -> dict[str, np.ndarray]:
        """Map Isaac observations into mini-pi0 canonical observation keys."""

        frame = self._image_from_raw(raw_obs, env_index=env_index)
        if frame is None:
            frame = np.zeros(
                (int(self.cfg.simulator.camera_height), int(self.cfg.simulator.camera_width), 3),
                dtype=np.uint8,
            )
        state_source = _flatten_named_arrays(raw_obs, env_index=env_index)
        state_source.update(self._scene_state(env_index))
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
        aliases["robot0_joint_pos"] = joint_pos if joint_pos is not None else _slice_or_zero(policy, 0, 9)
        aliases["robot0_joint_vel"] = joint_vel if joint_vel is not None else _slice_or_zero(policy, 9, 18)
        aliases["robot0_eef_pos"] = eef_pos if eef_pos is not None else np.zeros((3,), dtype=np.float32)
        aliases["robot0_eef_quat"] = eef_quat if eef_quat is not None else _quat_identity()
        aliases["robot0_gripper_qpos"] = gripper if gripper is not None else _slice_or_zero(policy, 7, 9)
        aliases["observation.state.object"] = obj if obj is not None else _slice_or_zero(policy, 18, 21)
        aliases["observation.state.task_progress"] = progress if progress is not None else np.zeros((1,), dtype=np.float32)
        aliases["observation.state.success"] = aliases["observation.state.task_progress"]
        aliases.update(state_source)
        return aliases

    def _scene_state(self, env_index: int) -> dict[str, np.ndarray]:
        """Read semantic robot and task state directly from the Isaac scene."""

        env = getattr(self, "env", None)
        if env is None:
            return {}
        unwrapped = getattr(env, "unwrapped", env)
        scene = getattr(unwrapped, "scene", None)
        if scene is None:
            return {}
        origin = _env_row(getattr(scene, "env_origins", None), env_index, width=3)
        if origin is None:
            origin = np.zeros(3, dtype=np.float32)
        ee_frame = _scene_entity(scene, "ee_frame")
        robot = _scene_entity(scene, "robot")
        obj = _scene_entity(scene, "object")
        state: dict[str, np.ndarray] = {}

        ee_data = getattr(ee_frame, "data", None)
        ee_pos_w = _env_row(getattr(ee_data, "target_pos_w", None), env_index, width=3)
        ee_quat_w = _env_row(getattr(ee_data, "target_quat_w", None), env_index, width=4)
        if ee_pos_w is not None:
            state["eef_pos"] = ee_pos_w - origin
        if ee_quat_w is not None:
            state["eef_quat"] = ee_quat_w

        robot_data = getattr(robot, "data", None)
        joint_pos = _env_row(getattr(robot_data, "joint_pos", None), env_index)
        if joint_pos is not None and joint_pos.size >= 2:
            state["gripper_pos"] = joint_pos[-2:]

        object_data = getattr(obj, "data", None)
        object_pos_w = _env_row(getattr(object_data, "root_pos_w", None), env_index, width=3)
        if object_pos_w is not None:
            state["object_pos"] = object_pos_w - origin
            state["task_progress"] = np.array(
                [self._lift_success(unwrapped, object_pos_w, origin, env_index)],
                dtype=np.float32,
            )
        return state

    @staticmethod
    def _lift_success(unwrapped: Any, object_pos_w: np.ndarray, origin: np.ndarray, env_index: int) -> float:
        """Return Franka lift completion from object height and goal distance."""

        manager = getattr(unwrapped, "command_manager", None)
        get_command = getattr(manager, "get_command", None)
        robot = _scene_entity(getattr(unwrapped, "scene", None), "robot")
        robot_pos_w = _env_row(getattr(getattr(robot, "data", None), "root_pos_w", None), env_index, width=3)
        if not callable(get_command) or robot_pos_w is None:
            return 0.0
        try:
            goal_local = _env_row(get_command("object_pose"), env_index)
        except (KeyError, RuntimeError):
            return 0.0
        if goal_local is None or goal_local.size < 3:
            return 0.0
        goal_pos_w = robot_pos_w + goal_local[:3]
        lifted = float(object_pos_w[2] - origin[2]) > 0.04
        return float(lifted and np.linalg.norm(object_pos_w - goal_pos_w) < 0.05)

    def _image_from_raw(self, raw_obs: Any, env_index: int = 0) -> np.ndarray | None:
        """Extract an RGB frame from nested Isaac observations."""

        for key in ("rgb", "image", "camera", "front", "agentview_image"):
            value = _find_nested(raw_obs, key)
            if value is None:
                continue
            arr = np.asarray(_to_numpy(value))
            if arr.ndim >= 4 and arr.shape[0] > env_index:
                arr = arr[env_index]
            if arr.ndim >= 3:
                return _as_uint8_rgb(arr)
        return self._image_from_sensor(env_index)

    def _image_from_sensor(self, env_index: int) -> np.ndarray | None:
        """Read one row from the configured tiled RGB camera."""

        unwrapped = getattr(self.env, "unwrapped", self.env)
        scene = getattr(unwrapped, "scene", None)
        sensors = getattr(scene, "sensors", {})
        sensor = sensors.get(_FRONT_CAMERA_NAME) if hasattr(sensors, "get") else None
        output = getattr(getattr(sensor, "data", None), "output", {})
        rgb = output.get("rgb") if hasattr(output, "get") else None
        if rgb is None:
            return None
        array = np.asarray(_to_numpy(rgb))
        if array.ndim >= 4:
            if not 0 <= env_index < array.shape[0]:
                return None
            array = array[env_index]
        return _as_uint8_rgb(array) if array.ndim >= 3 else None

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


def _flatten_named_arrays(value: Any, prefix: str = "", env_index: int = 0) -> dict[str, np.ndarray]:
    """Flatten nested mapping arrays into a name-to-array dictionary."""

    out: dict[str, np.ndarray] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_named_arrays(child, name, env_index=env_index))
        return out
    arr = np.asarray(_to_numpy(value), dtype=np.float32)
    if arr.size == 0:
        return out
    if arr.ndim >= 2:
        arr = arr[int(env_index)]
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


def _scene_entity(scene: Any, name: str) -> Any | None:
    """Return one named Isaac scene entity without assuming a concrete scene type."""

    if scene is None:
        return None
    try:
        return scene[name]
    except (KeyError, TypeError):
        return None


def _env_row(value: Any, env_index: int, *, width: int | None = None) -> np.ndarray | None:
    """Convert one environment row from an Isaac tensor to a flat NumPy array."""

    if value is None:
        return None
    array = np.asarray(_to_numpy(value), dtype=np.float32)
    if array.ndim >= 2:
        if not 0 <= env_index < array.shape[0]:
            return None
        array = array[env_index]
    flat = array.reshape(-1)
    if width is not None:
        if flat.size < width:
            return None
        flat = flat[:width]
    return flat.astype(np.float32, copy=False)


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
