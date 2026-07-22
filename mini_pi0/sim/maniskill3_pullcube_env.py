"""CUDA-vectorized PullCube task with per-scene dynamics randomization."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import sapien
import torch
from mani_skill.envs.tasks.tabletop.pull_cube_tool import PullCubeToolEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor

from mini_pi0.sim.pullcube_randomization import (
    parse_pullcube_domain_randomization,
)


@register_env("MiniPi0PullCubeToolDR-v1", max_episode_steps=450, override=True)
class MiniPi0PullCubeToolDREnv(PullCubeToolEnv):
    """PullCubeTool variant with strong initial-state and contact randomization.

    Pose randomization is resampled on every reset. GPU rigid-body mass and
    material properties cannot be changed after simulation initialization, so
    those are sampled once per vector sub-scene when the environment is built.
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch"]

    def __init__(
        self,
        *args: object,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.02,
        domain_randomization: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Create the task from validated randomization settings.

        Args:
            args: Positional arguments forwarded to ManiSkill.
            robot_uids: ManiSkill robot identifier.
            robot_init_qpos_noise: Nominal reset noise used when randomization is disabled.
            domain_randomization: Nested PullCube randomization configuration.
            kwargs: Additional ManiSkill environment arguments.
        """

        self.dr_config = parse_pullcube_domain_randomization(domain_randomization)
        qpos_noise = (
            self.dr_config.robot_init_qpos_noise
            if self.dr_config.enabled
            else float(robot_init_qpos_noise)
        )
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=qpos_noise,
            **kwargs,
        )

    def _load_agent(self, options: dict[str, object]) -> None:
        """Place Panda variants at their table-scene reset pose during construction."""

        if self.robot_uids in {"panda", "panda_wristcam"}:
            super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0.0]))
            return
        super()._load_agent(options)

    def _load_scene(self, options: dict[str, object]) -> None:
        """Build nominal actors or one randomized actor pair per CUDA sub-scene."""

        if not self.dr_config.enabled:
            super()._load_scene(options)
            return
        self.scene_builder = TableSceneBuilder(
            self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.scene_builder.build()
        cubes: list[Actor] = []
        tools: list[Actor] = []
        for scene_index in range(self.num_envs):
            rng = np.random.default_rng(self.dr_config.seed + 1009 * scene_index)
            cubes.append(self._build_cube(scene_index, rng))
            tools.append(self._build_tool(scene_index, rng))
        self.cube = self._merge(cubes, "cube")
        self.l_shape_tool = self._merge(tools, "l_shape_tool")

    def _build_cube(self, scene_index: int, rng: np.random.Generator) -> Actor:
        """Build one cube with scene-specific mass, material, and appearance."""

        config = self.dr_config
        mass_scale = float(rng.uniform(*config.cube_mass_scale_range))
        material = self._physics_material(rng)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=[self.cube_half_size] * 3,
            material=material,
            density=1000.0 * mass_scale,
        )
        builder.add_box_visual(
            half_size=[self.cube_half_size] * 3,
            material=self._render_material([12 / 255, 42 / 255, 160 / 255, 1.0], rng),
        )
        builder.set_scene_idxs([scene_index])
        builder.set_initial_pose(sapien.Pose(p=[0.0, 0.0, 0.5]))
        return builder.build(name=f"cube_{scene_index}")

    def _build_tool(self, scene_index: int, rng: np.random.Generator) -> Actor:
        """Build one L-shaped tool with scene-specific contact properties."""

        config = self.dr_config
        mass_scale = float(rng.uniform(*config.tool_mass_scale_range))
        material = self._physics_material(rng)
        render_material = self._render_material([1.0, 0.0, 0.0, 1.0], rng)
        builder = self.scene.create_actor_builder()
        handle_pose = sapien.Pose([self.handle_length / 2, 0.0, 0.0])
        hook_pose = sapien.Pose(
            [self.handle_length - self.hook_length / 2, self.width, 0.0]
        )
        builder.add_box_collision(
            handle_pose,
            [self.handle_length / 2, self.width / 2, self.height / 2],
            material=material,
            density=500.0 * mass_scale,
        )
        builder.add_box_visual(
            handle_pose,
            [self.handle_length / 2, self.width / 2, self.height / 2],
            material=render_material,
        )
        builder.add_box_collision(
            hook_pose,
            [self.hook_length / 2, self.width, self.height / 2],
            material=material,
            density=1000.0 * mass_scale,
        )
        builder.add_box_visual(
            hook_pose,
            [self.hook_length / 2, self.width, self.height / 2],
            material=render_material,
        )
        builder.set_scene_idxs([scene_index])
        builder.set_initial_pose(sapien.Pose(p=[0.0, 0.0, 0.5]))
        return builder.build(name=f"l_shape_tool_{scene_index}")

    def _merge(self, actors: list[Actor], name: str) -> Actor:
        """Merge per-scene actors into one batched ManiSkill actor view."""

        for actor in actors:
            self.remove_from_state_dict_registry(actor)
        merged = Actor.merge(actors, name=name)
        self.add_to_state_dict_registry(merged)
        return merged

    def _physics_material(self, rng: np.random.Generator) -> sapien.physx.PhysxMaterial:
        """Create one randomized contact material."""

        friction = float(rng.uniform(*self.dr_config.friction_range))
        restitution = float(rng.uniform(*self.dr_config.restitution_range))
        return sapien.pysapien.physx.PhysxMaterial(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=restitution,
        )

    def _render_material(
        self,
        base_color: list[float],
        rng: np.random.Generator,
    ) -> sapien.render.RenderMaterial:
        """Create a color-jittered visual material."""

        jitter = self.dr_config.color_jitter
        rgb = np.clip(
            np.asarray(base_color[:3], dtype=np.float32)
            + rng.uniform(-jitter, jitter, size=3),
            0.05,
            0.95,
        )
        material = sapien.render.RenderMaterial()
        material.set_base_color([*rgb.tolist(), float(base_color[3])])
        material.set_roughness(float(rng.uniform(0.55, 0.90)))
        material.set_metallic(float(rng.uniform(0.0, 0.25)))
        material.set_specular(float(rng.uniform(0.1, 0.4)))
        return material

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, object]) -> None:
        """Resample robot, tool, and cube initial state for selected environments."""

        if not self.dr_config.enabled:
            super()._initialize_episode(env_idx, options)
            return
        with torch.device(self.device):
            batch_size = len(env_idx)
            self.scene_builder.initialize(env_idx)
            tool_xyz = torch.zeros((batch_size, 3), device=self.device)
            tool_xyz[:, 0] = self._sample(env_idx, self.dr_config.tool_x_range)
            tool_xyz[:, 1] = self._sample(env_idx, self.dr_config.tool_y_range)
            tool_xyz[:, 2] = self.height / 2
            tool_quaternion = torch.zeros((batch_size, 4), device=self.device)
            tool_quaternion[:, 0] = 1.0
            self.l_shape_tool.set_pose(Pose.create_from_pq(tool_xyz, tool_quaternion))

            cube_xyz = torch.zeros((batch_size, 3), device=self.device)
            cube_xyz[:, 0] = self._sample(env_idx, self.dr_config.cube_x_range)
            cube_xyz[:, 1] = self._sample(env_idx, self.dr_config.cube_y_range)
            cube_xyz[:, 2] = self.cube_size / 2 + 0.015
            yaw_deg = self._sample(env_idx, self.dr_config.cube_yaw_range_deg)
            yaw = torch.deg2rad(yaw_deg)
            cube_quaternion = torch.zeros((batch_size, 4), device=self.device)
            cube_quaternion[:, 0] = torch.cos(0.5 * yaw)
            cube_quaternion[:, 3] = torch.sin(0.5 * yaw)
            self.cube.set_pose(Pose.create_from_pq(cube_xyz, cube_quaternion))

    def _sample(self, env_idx: torch.Tensor, value_range: tuple[float, float]) -> torch.Tensor:
        """Sample one deterministic scalar per selected environment."""

        values = self._batched_episode_rng[env_idx].uniform(*value_range)
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)
