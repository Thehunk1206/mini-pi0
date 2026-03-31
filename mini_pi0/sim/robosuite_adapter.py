from __future__ import annotations

from typing import Any

import numpy as np

from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys
from mini_pi0.sim.base import SimulatorAdapter, StepOutput


class RobosuiteAdapter(SimulatorAdapter):
    """Robosuite-backed implementation of the common simulator adapter API."""

    backend_name = "robosuite"

    def __init__(self, cfg: RootConfig):
        """Create a robosuite environment from typed config.

        Args:
            cfg: Root configuration object.
        """

        import robosuite as suite

        self.cfg = cfg
        self.suite = suite
        self._state_keys = effective_state_keys(cfg.robot)
        self._obs_alias_warned: set[tuple[str, str]] = set()
        self.env = suite.make(
            env_name=cfg.simulator.task,
            robots=cfg.simulator.robot,
            controller_configs=self._load_controller_configs(cfg.simulator.robot),
            has_renderer=cfg.simulator.has_renderer,
            has_offscreen_renderer=cfg.simulator.has_offscreen_renderer,
            use_camera_obs=cfg.simulator.use_camera_obs,
            camera_names=cfg.simulator.camera_names,
            camera_heights=cfg.simulator.camera_height,
            camera_widths=cfg.simulator.camera_width,
            control_freq=cfg.simulator.control_freq,
            horizon=cfg.simulator.horizon,
            reward_shaping=cfg.simulator.reward_shaping,
            ignore_done=False,
            seed=cfg.experiment.seed,
            **cfg.simulator.env_kwargs,
        )

    def _resolve_obs_key(self, obs: dict[str, Any], key: str) -> str:
        """Resolve configured observation key with robosuite-compatible aliases."""

        if key in obs:
            return key
        aliases = {
            "observation.images.base_0_rgb": "agentview_image",
            "observation.images.right_wrist_0_rgb": "robot0_eye_in_hand_image",
            "observation.images.wrist_0_rgb": "robot0_eye_in_hand_image",
            "observation.state.eef_pos": "robot0_eef_pos",
            "observation.state.eef_quat": "robot0_eef_quat",
            "observation.state.tool": "robot0_gripper_qpos",
            "observation.state.object": "object-state",
        }
        alt = aliases.get(key)
        if alt and alt in obs:
            pair = (key, alt)
            if pair not in self._obs_alias_warned:
                print(f"[robosuite] Obs key alias: '{key}' -> '{alt}'", flush=True)
                self._obs_alias_warned.add(pair)
            return alt
        available = ", ".join(sorted(obs.keys())[:20])
        raise KeyError(f"Observation key '{key}' not found. Available keys: {available}")

    def _load_controller_configs(self, robot: str = "Panda"):
        """Load controller config with compatibility across robosuite APIs.

        Args:
            robot: Robot name passed to composite config loader.

        Returns:
            Controller configuration object expected by ``suite.make``.

        Raises:
            RuntimeError: If no supported controller API is available.
        """

        suite = self.suite
        requested = str(self.cfg.simulator.controller).strip()
        requested_upper = requested.upper()
        single_arm_alias = {
            "BASIC": "OSC_POSE",
            "OSC_POSE": "OSC_POSE",
            "OSC_POSITION": "OSC_POSITION",
            "JOINT_POSITION": "JOINT_POSITION",
            "JOINT_VELOCITY": "JOINT_VELOCITY",
            "IK_POSE": "IK_POSE",
        }
        if hasattr(suite, "load_controller_config"):
            tried: list[str] = []
            candidates = []
            if requested:
                candidates.append(requested)
            mapped = single_arm_alias.get(requested_upper)
            if mapped and mapped not in candidates:
                candidates.append(mapped)
            if "OSC_POSE" not in candidates:
                candidates.append("OSC_POSE")

            for cand in candidates:
                tried.append(cand)
                try:
                    out = suite.load_controller_config(default_controller=cand)
                    print(
                        f"[robosuite] controller_config=load_controller_config default_controller={cand} "
                        f"(requested={requested or 'none'})",
                        flush=True,
                    )
                    return out
                except Exception:
                    continue
            raise RuntimeError(
                "Failed to load robosuite single-arm controller config. "
                f"requested={requested!r}, tried={tried}"
            )
        if hasattr(suite, "load_composite_controller_config"):
            cfg = suite.load_composite_controller_config(controller=self.cfg.simulator.controller, robot=robot)
            body_parts = cfg.get("body_parts", None)
            if isinstance(body_parts, dict):
                preferred = [k for k in ("right", "arm", "single") if k in body_parts]
                if preferred:
                    keep = set(preferred)
                    cfg["body_parts"] = {k: v for k, v in body_parts.items() if k in keep}
            print(
                f"[robosuite] controller_config=load_composite_controller_config "
                f"controller={self.cfg.simulator.controller} robot={robot}",
                flush=True,
            )
            return cfg
        raise RuntimeError("Unsupported robosuite controller API")

    def _canon_obs(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Convert raw robosuite observation dict into canonical schema.

        Args:
            obs: Robosuite observation dictionary.

        Returns:
            Canonical observation dictionary with configured keys/dtypes.
        """

        out: dict[str, np.ndarray] = {}
        for image_key in effective_image_keys(self.cfg.robot):
            image_src = self._resolve_obs_key(obs, image_key)
            out[image_key] = np.asarray(obs[image_src], dtype=np.uint8)
        for key in self._state_keys:
            src = self._resolve_obs_key(obs, key)
            out[key] = np.asarray(obs[src], dtype=np.float32)
        return out

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset robosuite environment and return canonical observation.

        Args:
            seed: Unused (robosuite seed is configured at environment creation time).

        Returns:
            Canonical observation dictionary.
        """

        # robosuite's reset is seeded at make-time for this adapter.
        obs = self.env.reset()
        return self._canon_obs(obs)

    def step(self, action: np.ndarray) -> StepOutput:
        """Execute one clipped action in robosuite.

        Args:
            action: Action vector in environment action space.

        Returns:
            Canonical step output.
        """

        lo, hi = self.action_spec()
        clipped = np.clip(np.asarray(action, dtype=np.float32), lo, hi)
        obs, reward, done, info = self.env.step(clipped)
        return StepOutput(self._canon_obs(obs), float(reward), bool(done), dict(info))

    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        """Return robosuite action bounds.

        Returns:
            Tuple of lower/upper action limits.
        """

        lo, hi = self.env.action_spec
        return np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32)

    def render(self, camera: str = "agentview", width: int = 512, height: int = 512) -> np.ndarray:
        """Render a frame from robosuite simulation.

        Args:
            camera: Camera name.
            width: Output width.
            height: Output height.

        Returns:
            Rendered frame as ``uint8`` array when possible.
        """

        try:
            return np.asarray(self.env.sim.render(height=height, width=width, camera_name=camera), dtype=np.uint8)
        except TypeError:
            return np.asarray(self.env.sim.render(width, height), dtype=np.uint8)

    def check_success(self, info: dict[str, Any] | None = None, obs: dict[str, np.ndarray] | None = None) -> bool:
        """Determine task success using info dict or robosuite internal checks.

        Args:
            info: Optional step info dict.
            obs: Unused canonical observation (kept for interface parity).

        Returns:
            ``True`` if success criteria are satisfied.
        """

        if info is not None and bool(info.get("success", False)):
            return True
        if hasattr(self.env, "_check_success"):
            try:
                return bool(self.env._check_success())
            except Exception:
                return False
        return False

    def set_object_pose(
        self,
        object_name: str = "cube",
        xy: tuple[float, float] | None = None,
        xy_range: tuple[float, float, float, float] | None = None,
        z: float | None = None,
        yaw_deg: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> bool:
        """Override Lift cube pose for controlled evaluation randomization.

        Args:
            object_name: Supported object semantic name (currently ``cube``).
            xy: Optional fixed ``(x, y)`` position.
            xy_range: Optional random range ``(xmin, xmax, ymin, ymax)``.
            z: Optional fixed z position.
            yaw_deg: Optional fixed yaw in degrees.
            rng: Optional random generator used with ``xy_range``.

        Returns:
            ``True`` when pose was applied, otherwise ``False``.
        """

        if object_name.lower() != "cube":
            return False
        joint_name = "cube_joint0"
        try:
            qpos = np.asarray(self.env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
        except Exception:
            return False

        if xy is not None:
            qpos[0], qpos[1] = float(xy[0]), float(xy[1])
        elif xy_range is not None:
            xmin, xmax, ymin, ymax = [float(v) for v in xy_range]
            gen = np.random.default_rng() if rng is None else rng
            qpos[0] = float(gen.uniform(xmin, xmax))
            qpos[1] = float(gen.uniform(ymin, ymax))

        if z is not None:
            qpos[2] = float(z)
        if yaw_deg is not None:
            yaw = np.deg2rad(float(yaw_deg))
            qpos[3:7] = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)

        self.env.sim.data.set_joint_qpos(joint_name, qpos)
        try:
            qvel = np.asarray(self.env.sim.data.get_joint_qvel(joint_name), dtype=np.float64)
            self.env.sim.data.set_joint_qvel(joint_name, np.zeros_like(qvel))
        except Exception:
            pass
        self.env.sim.forward()
        return True

    def close(self) -> None:
        """Close robosuite environment and release rendering resources."""

        self.env.close()
