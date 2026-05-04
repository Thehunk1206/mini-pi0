from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import sapien
import torch
from transforms3d import euler
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs import Pose

from mini_pi0.sim.domain_randomization import DomainRandomizationConfig, parse_domain_randomization_config


@dataclass
class RewardWeights:
    progress: float = 8.0
    place_bonus: float = 4.0
    terminal_success: float = 12.0
    terminal_timeout: float = 2.0
    shaping_eef_to_obj: float = 0.25
    shaping_obj_to_tray: float = 0.35
    shaping_lift: float = 0.15
    penalty_drop: float = 0.8
    penalty_collision: float = 0.1
    penalty_out_of_workspace: float = 0.6
    step_cost: float = 0.01


def _render_material(
    color: list[float],
    *,
    roughness: float = 0.85,
    metallic: float = 0.0,
    specular: float = 0.25,
) -> sapien.render.RenderMaterial:
    """Create a SAPIEN material with stable PBR-like parameters."""
    mat = sapien.render.RenderMaterial()
    mat.set_base_color(color)
    mat.set_roughness(float(np.clip(roughness, 0.0, 1.0)))
    mat.set_metallic(float(np.clip(metallic, 0.0, 1.0)))
    mat.set_specular(float(np.clip(specular, 0.0, 1.0)))
    return mat


def _build_material_box(
    scene,
    *,
    half_sizes: list[float],
    material: sapien.render.RenderMaterial,
    name: str,
    body_type: str,
    add_collision: bool,
    initial_pose: sapien.Pose,
):
    """Build a box actor using a configured render material."""
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(half_size=half_sizes)
    builder.add_box_visual(half_size=half_sizes, material=material)
    if body_type == "dynamic":
        builder.set_initial_pose(initial_pose)
        return builder.build(name=name)
    if body_type == "kinematic":
        builder.set_initial_pose(initial_pose)
        return builder.build_kinematic(name=name)
    if body_type == "static":
        builder.set_initial_pose(initial_pose)
        return builder.build_static(name=name)
    raise ValueError(f"Unknown body type {body_type}")


@register_env("MiniPi0MultiObjectTray-v1", max_episode_steps=1000, override=True)
class MiniPi0MultiObjectTrayEnv(BaseEnv):
    """Custom ManiSkill task: pick random objects and place all in tray.

    Object shapes: cube, sphere, and cone-as-pyramid proxy (implemented as
    short cylinder for physics stability and simple primitive assets).
    """

    SUPPORTED_ROBOTS = ["panda", "fetch", "so100", "widowxai", "xarm6_robotiq"]
    SUPPORTED_REWARD_MODES = ["dense", "sparse", "none", "normalized_dense"]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        object_count_min: int = 1,
        object_count_max: int = 10,
        object_types: tuple[str, ...] = ("cube", "sphere", "cone"),
        tray_center=(0.20, 0.22, 0.006),
        tray_size_xy=(0.22, 0.22),
        tray_wall_thickness: float = 0.008,
        tray_wall_height: float = 0.035,
        bowl_center=(0.20, -0.22, 0.006),
        bowl_inner_radius: float = 0.11,
        bowl_spawn_height: float = 0.0,
        bowl_table_z_offset: float = 0.02,
        bowl_pose_quat=(1.0, 0.0, 0.0, 0.0),
        bowl_spawn_plane_offset: float = 0.035,
        source_bowl_collision: bool = True,
        source_bowl_containment: bool = True,
        source_bowl_wall_segments: int = 16,
        source_bowl_wall_thickness: float = 0.008,
        source_bowl_wall_height: float = 0.04,
        source_bowl_wall_radius_offset: float = 0.025,
        scripted_grasp_assist: bool = False,
        grasp_assist_radius: float = 0.085,
        grasp_assist_z_offset: float = -0.04,
        reset_settle_steps: int = 18,
        tray_z_tol: float = 0.03,
        tray_place_min_z: float = 0.012,
        tray_place_max_z: float = 0.14,
        settle_steps_required: int = 3,
        settle_speed_threshold: float = 0.03,
        lift_height_threshold: float = 0.06,
        spawn_min=(-0.10, -0.22),
        spawn_max=(0.10, 0.22),
        sensor_width: int = 128,
        sensor_height: int = 128,
        render_width: int = 512,
        render_height: int = 512,
        timeout_steps: int = 1000,
        reward: dict[str, float] | None = None,
        domain_randomization: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.dr_config: DomainRandomizationConfig = parse_domain_randomization_config(domain_randomization)
        self.domain_randomization = bool(self.dr_config.enabled)
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self.object_count_min = int(max(1, object_count_min))
        self.object_count_max = int(max(self.object_count_min, object_count_max))
        self.object_types = tuple(x for x in object_types if x in {"cube", "sphere", "cone"}) or ("cube", "sphere", "cone")

        self.tray_center_np = np.asarray(tray_center, dtype=np.float32)
        self.tray_size_xy_np = np.asarray(tray_size_xy, dtype=np.float32)
        self.tray_wall_thickness = float(max(0.004, tray_wall_thickness))
        self.tray_wall_height = float(max(0.015, tray_wall_height))
        self.bowl_center_np = np.asarray(bowl_center, dtype=np.float32)
        self.bowl_inner_radius = float(max(0.04, bowl_inner_radius))
        self.bowl_spawn_height = float(max(0.0, bowl_spawn_height))
        self.bowl_table_z_offset = float(max(0.0, bowl_table_z_offset))
        self.bowl_pose_quat = np.asarray(bowl_pose_quat, dtype=np.float32)
        self.bowl_spawn_plane_offset = float(max(0.0, bowl_spawn_plane_offset))
        self.source_bowl_collision = bool(source_bowl_collision)
        self.source_bowl_containment = bool(source_bowl_containment)
        self.source_bowl_wall_segments = int(max(8, source_bowl_wall_segments))
        self.source_bowl_wall_thickness = float(max(0.004, source_bowl_wall_thickness))
        self.source_bowl_wall_height = float(max(0.015, source_bowl_wall_height))
        self.source_bowl_wall_radius_offset = float(max(0.0, source_bowl_wall_radius_offset))
        self.scripted_grasp_assist = bool(scripted_grasp_assist)
        self.grasp_assist_radius = float(max(0.0, grasp_assist_radius))
        self.grasp_assist_z_offset = float(grasp_assist_z_offset)
        self.reset_settle_steps = int(max(0, reset_settle_steps))
        self.tray_z_tol = float(tray_z_tol)
        self.tray_place_min_z = float(tray_place_min_z)
        self.tray_place_max_z = float(max(tray_place_min_z, tray_place_max_z))
        self.settle_steps_required = int(max(1, settle_steps_required))
        self.settle_speed_threshold = float(settle_speed_threshold)
        self.lift_height_threshold = float(lift_height_threshold)
        self.spawn_min = np.asarray(spawn_min, dtype=np.float32)
        self.spawn_max = np.asarray(spawn_max, dtype=np.float32)
        self.sensor_width = int(max(64, sensor_width))
        self.sensor_height = int(max(64, sensor_height))
        self.render_width = int(max(128, render_width))
        self.render_height = int(max(128, render_height))
        self.timeout_steps = int(max(1, timeout_steps))

        self.weights = RewardWeights()
        for k, v in (reward or {}).items():
            if hasattr(self.weights, k):
                setattr(self.weights, k, float(v))

        self._active_object_mask = None
        self._placed_mask = None
        self._placed_counted_mask = None
        self._settle_counter = None
        self._prev_success_fraction = None
        self._last_success_fraction = None
        self._last_held_mask = None
        self._grasp_assist_held_idx = None
        self._target_idx = None
        self._placement_targets = None
        self._last_reward_terms: dict[str, torch.Tensor] | None = None

        robot_key = str(robot_uids).strip().lower()
        cfg = PICK_CUBE_CONFIGS.get(robot_key, PICK_CUBE_CONFIGS["panda"])
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        cam_cfg = self.dr_config.camera
        base_eye = np.asarray(self.sensor_cam_eye_pos, dtype=np.float32).copy()
        base_target = np.asarray(self.sensor_cam_target_pos, dtype=np.float32).copy()
        hand_p = np.array([0.0464982, -0.0200011, 0.0360011], dtype=np.float32)
        fov = np.pi / 2
        if self.domain_randomization and cam_cfg.enabled:
            base_eye += self._np_uniform(-np.asarray(cam_cfg.base_pos_jitter), np.asarray(cam_cfg.base_pos_jitter))
            base_target += self._np_uniform(-np.asarray(cam_cfg.base_target_jitter), np.asarray(cam_cfg.base_target_jitter))
            hand_p += self._np_uniform(-np.asarray(cam_cfg.hand_pos_jitter), np.asarray(cam_cfg.hand_pos_jitter))
            fov += np.deg2rad(float(self._np_uniform(-cam_cfg.fov_jitter_deg, cam_cfg.fov_jitter_deg)))

        base_pose = sapien_utils.look_at(eye=base_eye, target=base_target)
        hand_mount = self.agent.robot.links_map.get("panda_hand")
        hand_pose = sapien.Pose(
            p=hand_p.tolist(),
            q=[0.0, 0.70710678, 0.0, 0.70710678],
        )
        configs = [CameraConfig("base_camera", base_pose, self.sensor_width, self.sensor_height, fov, 0.01, 100, shader_pack="default")]
        if hand_mount is not None:
            configs.append(
                CameraConfig(
                    "hand_camera",
                    hand_pose,
                    self.sensor_width,
                    self.sensor_height,
                    fov,
                    0.01,
                    100,
                    mount=hand_mount,
                    shader_pack="default",
                )
            )
        return configs

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=self.human_cam_eye_pos, target=self.human_cam_target_pos)
        return CameraConfig("render_camera", pose, self.render_width, self.render_height, 1, 0.01, 100)

    def _load_lighting(self, options: dict):
        """Load default or randomized scene lighting."""
        if not (self.domain_randomization and self.dr_config.lighting.enabled):
            return super()._load_lighting(options)

        light_cfg = self.dr_config.lighting
        ambient = float(self._np_uniform(*light_cfg.ambient_range))
        intensity = float(self._np_uniform(*light_cfg.directional_intensity_range))
        yaw = np.deg2rad(float(self._np_uniform(*light_cfg.directional_yaw_range_deg)))
        direction = [float(np.cos(yaw)), float(np.sin(yaw)), -1.0]
        self.scene.set_ambient_light([ambient, ambient, ambient])
        self.scene.add_directional_light(
            direction,
            [intensity, intensity, intensity],
            shadow=self.enable_shadow,
            shadow_scale=5,
            shadow_map_size=2048,
        )
        self.scene.add_directional_light([-0.5, -0.25, -1], [0.45, 0.42, 0.38])
        self.scene.add_directional_light([0.25, 0.65, -1], [0.25, 0.27, 0.30])

    def _np_uniform(self, low: Any, high: Any) -> np.ndarray | float:
        """Sample from the environment RNG if available, otherwise numpy RNG."""
        rng = getattr(self, "_batched_episode_rng", None)
        if rng is not None:
            try:
                return rng[0].uniform(low=low, high=high)
            except Exception:
                return rng.uniform(low=low, high=high)
        return np.random.default_rng().uniform(low=low, high=high)

    @staticmethod
    def _jitter_color(base: list[float], jitter: float, rng: np.random.Generator) -> list[float]:
        """Apply bounded RGB jitter while preserving alpha."""
        rgb = np.asarray(base[:3], dtype=np.float32)
        if jitter > 0.0:
            rgb = np.clip(rgb + rng.uniform(-jitter, jitter, size=3), 0.05, 0.95)
        return rgb.tolist() + [float(base[3] if len(base) > 3 else 1.0)]

    @staticmethod
    def _material_for_shape(shape: str, color: list[float], rng: np.random.Generator | None = None) -> sapien.render.RenderMaterial:
        """Return muted object material presets by shape."""
        roughness = 0.72
        specular = 0.18
        if rng is not None:
            roughness = float(np.clip(roughness + rng.uniform(-0.08, 0.08), 0.55, 0.9))
            specular = float(np.clip(specular + rng.uniform(-0.06, 0.06), 0.08, 0.35))
        if shape == "sphere":
            roughness = min(0.82, roughness + 0.05)
            specular = min(0.35, specular + 0.08)
        return _render_material(color, roughness=roughness, metallic=0.0, specular=specular)

    def _build_randomized_object(self, idx: int, shape: str, base_color: list[float]) -> Any:
        """Build one logical object, optionally merged from per-subscene variants."""
        if not self.domain_randomization:
            return self._build_shared_object(idx, shape, base_color)

        obj_cfg = self.dr_config.visual
        phys_cfg = self.dr_config.physics
        built = []
        for scene_i in range(self.num_envs):
            rng = self._batched_episode_rng[scene_i]
            color = self._jitter_color(base_color, obj_cfg.object_color_jitter if obj_cfg.enabled else 0.0, rng)
            friction = float(rng.uniform(*phys_cfg.object_friction_range)) if phys_cfg.enabled else 1.0
            restitution = float(rng.uniform(*phys_cfg.object_restitution_range)) if phys_cfg.enabled else 0.0
            mass_scale = float(rng.uniform(*phys_cfg.object_mass_scale_range)) if phys_cfg.enabled else 1.0
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=restitution,
            )
            builder = self.scene.create_actor_builder()
            render_mat = self._material_for_shape(shape, color, rng)
            if shape == "cube":
                builder.add_box_collision(half_size=[0.02] * 3, material=material, density=1000 * mass_scale)
                builder.add_box_visual(half_size=[0.02] * 3, material=render_mat)
            elif shape == "sphere":
                builder.add_sphere_collision(radius=0.022, material=material, density=1000 * mass_scale)
                builder.add_sphere_visual(radius=0.022, material=render_mat)
            else:
                builder.add_cylinder_collision(radius=0.020, half_length=0.020, material=material, density=1000 * mass_scale)
                builder.add_cylinder_visual(radius=0.020, half_length=0.020, material=render_mat)
            builder.set_scene_idxs([scene_i])
            builder.initial_pose = sapien.Pose(p=[5, 5, 5])
            actor = builder.build(name=f"obj_{idx:02d}_{shape}_{scene_i}")
            self.remove_from_state_dict_registry(actor)
            built.append(actor)
        merged = Actor.merge(built, name=f"obj_{idx:02d}_{shape}")
        self.add_to_state_dict_registry(merged)
        return merged

    def _build_shared_object(self, idx: int, shape: str, color: list[float]) -> Any:
        """Build one shared primitive object for non-randomized runs."""
        name = f"obj_{idx:02d}_{shape}"
        material = self._material_for_shape(shape, color)
        builder = self.scene.create_actor_builder()
        if shape == "cube":
            builder.add_box_collision(half_size=[0.02] * 3)
            builder.add_box_visual(half_size=[0.02] * 3, material=material)
        elif shape == "sphere":
            builder.add_sphere_collision(radius=0.022)
            builder.add_sphere_visual(radius=0.022, material=material)
        else:
            builder.add_cylinder_collision(radius=0.020, half_length=0.020)
            builder.add_cylinder_visual(radius=0.020, half_length=0.020, material=material)
        builder.set_initial_pose(sapien.Pose(p=[5, 5, 5]))
        return builder.build(name=name)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        self.objects: list[Any] = []
        self.object_shape_names: list[str] = []
        base_colors = {
            "cube": [0.64, 0.26, 0.22, 1.0],
            "sphere": [0.20, 0.43, 0.58, 1.0],
            "cone": [0.68, 0.59, 0.30, 1.0],
        }

        for idx in range(self.object_count_max):
            shape = self.object_types[idx % len(self.object_types)]
            actor = self._build_randomized_object(idx, shape, base_colors[shape])
            self.objects.append(actor)
            self.object_shape_names.append(shape)

        tray_base_half_z = 0.006
        sx = float(self.tray_size_xy_np[0]) * 0.5
        sy = float(self.tray_size_xy_np[1]) * 0.5
        wt = self.tray_wall_thickness
        wh = self.tray_wall_height
        wall_half_h = wh * 0.5
        base_z = float(self.tray_center_np[2])
        rng0 = self._batched_episode_rng[0] if self.domain_randomization else np.random.default_rng(0)
        tray_color = [0.31, 0.33, 0.34, 1.0]
        tray_wall_color = [0.22, 0.23, 0.24, 1.0]
        bowl_wall_color = [0.40, 0.39, 0.36, 0.0]
        bowl_color = [0.70, 0.67, 0.61, 1.0]
        if self.domain_randomization and self.dr_config.visual.enabled:
            tray_color = self._jitter_color(tray_color, self.dr_config.visual.tray_color_jitter, rng0)
            tray_wall_color = self._jitter_color(tray_wall_color, self.dr_config.visual.tray_color_jitter, rng0)
            bowl_wall_color = self._jitter_color(bowl_wall_color, self.dr_config.visual.bowl_color_jitter, rng0)
            bowl_color = self._jitter_color(bowl_color, self.dr_config.visual.bowl_color_jitter, rng0)

        self.tray = _build_material_box(
            self.scene,
            half_sizes=[sx, sy, tray_base_half_z],
            material=_render_material(tray_color, roughness=0.78, metallic=0.05, specular=0.20),
            name="tray_base",
            body_type="kinematic",
            add_collision=True,
            initial_pose=sapien.Pose(p=self.tray_center_np.tolist()),
        )
        self.tray_walls = []
        wall_specs = [
            ([sx + wt * 0.5, 0.0, tray_base_half_z + wall_half_h], [wt * 0.5, sy + wt, wall_half_h], "tray_wall_pos_x"),
            ([-sx - wt * 0.5, 0.0, tray_base_half_z + wall_half_h], [wt * 0.5, sy + wt, wall_half_h], "tray_wall_neg_x"),
            ([0.0, sy + wt * 0.5, tray_base_half_z + wall_half_h], [sx + wt, wt * 0.5, wall_half_h], "tray_wall_pos_y"),
            ([0.0, -sy - wt * 0.5, tray_base_half_z + wall_half_h], [sx + wt, wt * 0.5, wall_half_h], "tray_wall_neg_y"),
        ]
        for rel_p, hs, nm in wall_specs:
            p = (self.tray_center_np + np.asarray(rel_p, dtype=np.float32)).tolist()
            wall = _build_material_box(
                self.scene,
                half_sizes=hs,
                material=_render_material(tray_wall_color, roughness=0.72, metallic=0.08, specular=0.18),
                name=nm,
                body_type="kinematic",
                add_collision=True,
                initial_pose=sapien.Pose(p=p),
            )
            self.tray_walls.append(wall)

        # Source bowl on right side (-y): use ManiSkill humanoid-task bowl mesh pattern.
        # This mesh needs a fixed +90deg X corrective pose to be upright.
        bowl_builder = self.scene.create_actor_builder()
        bowl_fix_q = euler.euler2quat(np.pi / 2, 0, 0)
        import mani_skill.envs.tasks.humanoid.humanoid_pick_place as humanoid_pick_place
        bowl_assets_dir = os.path.join(os.path.dirname(humanoid_pick_place.__file__), "assets")
        if self.source_bowl_collision:
            bowl_builder.add_nonconvex_collision_from_file(
                filename=os.path.join(bowl_assets_dir, "frl_apartment_bowl_07.ply"),
                pose=sapien.Pose(q=bowl_fix_q),
                scale=[1.0, 1.0, 1.0],
            )
        bowl_builder.add_visual_from_file(
            filename=os.path.join(bowl_assets_dir, "frl_apartment_bowl_07.glb"),
            scale=[1.0, 1.0, 1.0],
            pose=sapien.Pose(q=bowl_fix_q),
            material=_render_material(bowl_color, roughness=0.82, metallic=0.0, specular=0.22),
        )
        bowl_pose_p = self.bowl_center_np.copy()
        bowl_pose_p[2] += self.bowl_table_z_offset
        bowl_builder.initial_pose = sapien.Pose(p=bowl_pose_p.tolist(), q=self.bowl_pose_quat.tolist())
        self.bowl = bowl_builder.build_kinematic(name="source_bowl")
        self.bowl_containment_walls = []
        if self.source_bowl_containment:
            radius = max(0.03, self.bowl_inner_radius + self.source_bowl_wall_radius_offset)
            segment_half_len = radius * np.tan(np.pi / self.source_bowl_wall_segments) * 1.15
            wall_half_h = self.source_bowl_wall_height * 0.5
            wall_z = float(self.bowl_center_np[2] + self.bowl_table_z_offset + wall_half_h)
            for wall_idx in range(self.source_bowl_wall_segments):
                theta = 2.0 * np.pi * wall_idx / self.source_bowl_wall_segments
                center = np.array(
                    [
                        self.bowl_center_np[0] + radius * np.cos(theta),
                        self.bowl_center_np[1] + radius * np.sin(theta),
                        wall_z,
                    ],
                    dtype=np.float32,
                )
                wall = _build_material_box(
                    self.scene,
                    half_sizes=[self.source_bowl_wall_thickness * 0.5, segment_half_len, wall_half_h],
                    material=_render_material(bowl_wall_color, roughness=0.86, metallic=0.0, specular=0.12),
                    name=f"source_bowl_containment_{wall_idx:02d}",
                    body_type="kinematic",
                    add_collision=True,
                    initial_pose=sapien.Pose(p=center.tolist(), q=euler.euler2quat(0, 0, theta).tolist()),
                )
                self.bowl_containment_walls.append(wall)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self._active_object_mask = torch.zeros((self.num_envs, self.object_count_max), dtype=torch.bool, device=self.device)
            self._placed_mask = torch.zeros_like(self._active_object_mask)
            self._placed_counted_mask = torch.zeros_like(self._active_object_mask)
            self._settle_counter = torch.zeros((self.num_envs, self.object_count_max), dtype=torch.int32, device=self.device)
            self._prev_success_fraction = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            self._last_success_fraction = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            self._grasp_assist_held_idx = torch.full((self.num_envs,), -1, dtype=torch.int64, device=self.device)
            self._target_idx = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
            self._placement_targets = self._sample_placement_targets()
            self._last_reward_terms = None

            active_counts = torch.randint(
                low=self.object_count_min,
                high=self.object_count_max + 1,
                size=(self.num_envs,),
                device=self.device,
            )
            if self.domain_randomization and self.dr_config.objects.enabled and self.dr_config.objects.randomize_active_slots:
                for env_i in range(self.num_envs):
                    perm = torch.randperm(self.object_count_max, device=self.device)
                    self._active_object_mask[env_i, perm[: int(active_counts[env_i].item())]] = True
            else:
                for i in range(self.object_count_max):
                    self._active_object_mask[:, i] = i < active_counts

            # Spawn active objects inside bowl with simple separation retries.
            device = self.device
            bowl_center_xy = torch.tensor(self.bowl_center_np[:2], dtype=torch.float32, device=device)
            spawn_margin = max(0.03, self.source_bowl_wall_thickness + 0.026)
            base_spawn_r = max(0.02, self.bowl_inner_radius - spawn_margin)
            spawn_jitter = self.dr_config.objects.spawn_radius_jitter if self.domain_randomization and self.dr_config.objects.enabled else 0.0
            min_sep = 0.045
            xyz_store: list[torch.Tensor] = []
            for i, actor in enumerate(self.objects):
                xy = torch.zeros((self.num_envs, 2), device=device)
                for env_i in range(self.num_envs):
                    spawn_r = base_spawn_r
                    if spawn_jitter > 0.0:
                        spawn_r = max(0.02, base_spawn_r + float(torch.empty((), device=device).uniform_(-spawn_jitter, spawn_jitter).item()))
                    best_xy = None
                    for _ in range(25):
                        theta = float(torch.rand((1,), device=device).item()) * 2.0 * np.pi
                        r = spawn_r * np.sqrt(float(torch.rand((1,), device=device).item()))
                        cand = bowl_center_xy + torch.tensor([r * np.cos(theta), r * np.sin(theta)], device=device)
                        valid = True
                        for prev in xyz_store:
                            d = torch.linalg.norm(cand - prev[env_i, :2]).item()
                            if d < min_sep:
                                valid = False
                                break
                        if valid:
                            best_xy = cand
                            break
                    if best_xy is None:
                        # fallback sample without separation constraint
                        theta = float(torch.rand((1,), device=device).item()) * 2.0 * np.pi
                        r = spawn_r * np.sqrt(float(torch.rand((1,), device=device).item()))
                        best_xy = bowl_center_xy + torch.tensor([r * np.cos(theta), r * np.sin(theta)], device=device)
                    xy[env_i] = best_xy

                xyz = torch.zeros((self.num_envs, 3), device=device)
                xyz[:, :2] = xy
                if self.object_shape_names[i] == "sphere":
                    z_offset = 0.022
                elif self.object_shape_names[i] == "cube":
                    z_offset = 0.020
                else:
                    z_offset = 0.020
                xyz[:, 2] = (
                    float(self.bowl_center_np[2] + self.bowl_table_z_offset)
                    + self.bowl_spawn_plane_offset
                    + z_offset
                    + self.bowl_spawn_height
                )
                qs = torch.zeros((self.num_envs, 4), device=self.device)
                qs[:, 0] = 1.0
                if self.domain_randomization and self.dr_config.objects.enabled and self.dr_config.objects.randomize_spawn_yaw and self.object_shape_names[i] != "sphere":
                    yaw = torch.rand((self.num_envs,), device=self.device) * (2.0 * np.pi)
                    qs[:, 0] = torch.cos(yaw * 0.5)
                    qs[:, 3] = torch.sin(yaw * 0.5)
                actor.set_pose(Pose.create_from_pq(p=xyz, q=qs))
                xyz_store.append(xyz)

                inactive = ~self._active_object_mask[:, i]
                if torch.any(inactive):
                    hide_xyz = xyz.clone()
                    hide_xyz[inactive] = torch.tensor([5.0, 5.0, 5.0], device=self.device)
                    actor.set_pose(Pose.create_from_pq(p=hide_xyz, q=qs))

            self.tray.set_pose(
                Pose.create_from_pq(
                    p=torch.tensor(self.tray_center_np, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                    q=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                )
            )
            sx = float(self.tray_size_xy_np[0]) * 0.5
            sy = float(self.tray_size_xy_np[1]) * 0.5
            wt = self.tray_wall_thickness
            wh = self.tray_wall_height
            wall_half_h = wh * 0.5
            rel_poses = [
                [sx + wt * 0.5, 0.0, 0.006 + wall_half_h],
                [-sx - wt * 0.5, 0.0, 0.006 + wall_half_h],
                [0.0, sy + wt * 0.5, 0.006 + wall_half_h],
                [0.0, -sy - wt * 0.5, 0.006 + wall_half_h],
            ]
            for wall, rel_p in zip(self.tray_walls, rel_poses):
                p = torch.tensor(self.tray_center_np + np.asarray(rel_p, dtype=np.float32), dtype=torch.float32, device=self.device)
                p = p.unsqueeze(0).repeat(self.num_envs, 1)
                q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
                wall.set_pose(Pose.create_from_pq(p=p, q=q))

            # Reset bowl to configured right-side position.
            bowl_pose_p = self.bowl_center_np.copy()
            bowl_pose_p[2] += self.bowl_table_z_offset
            self.bowl.set_pose(
                Pose.create_from_pq(
                    p=torch.tensor(bowl_pose_p, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                    q=torch.tensor(self.bowl_pose_quat, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                )
            )
            if self.source_bowl_containment:
                radius = max(0.03, self.bowl_inner_radius + self.source_bowl_wall_radius_offset)
                wall_half_h = self.source_bowl_wall_height * 0.5
                wall_z = float(self.bowl_center_np[2] + self.bowl_table_z_offset + wall_half_h)
                for wall_idx, wall in enumerate(self.bowl_containment_walls):
                    theta = 2.0 * np.pi * wall_idx / self.source_bowl_wall_segments
                    p = torch.tensor(
                        [
                            self.bowl_center_np[0] + radius * np.cos(theta),
                            self.bowl_center_np[1] + radius * np.sin(theta),
                            wall_z,
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0).repeat(self.num_envs, 1)
                    q = torch.tensor(euler.euler2quat(0, 0, theta), dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
                    wall.set_pose(Pose.create_from_pq(p=p, q=q))
            robot_pose_before_settle = self.agent.robot.pose
            robot_qpos_before_settle = self.agent.robot.get_qpos().clone()
            if self.reset_settle_steps > 0:
                for _ in range(self.reset_settle_steps):
                    self.scene.step()
                self.agent.reset(robot_qpos_before_settle)
                self.agent.robot.set_pose(robot_pose_before_settle)
            for actor in self.objects:
                try:
                    actor.set_linear_velocity(torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device))
                    actor.set_angular_velocity(torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device))
                except Exception:
                    pass

    def _get_object_pos_tensor(self) -> torch.Tensor:
        pos = []
        for actor in self.objects:
            pos.append(actor.pose.p)
        return torch.stack(pos, dim=1)

    def _sample_placement_targets(self) -> torch.Tensor:
        """Sample per-object drop targets inside the tray."""
        targets = torch.zeros((self.num_envs, self.object_count_max, 3), dtype=torch.float32, device=self.device)
        tray_center = torch.tensor(self.tray_center_np, dtype=torch.float32, device=self.device)
        if not (self.domain_randomization and self.dr_config.placement.enabled):
            targets[:] = tray_center
            targets[..., 2] = float(self.dr_config.placement.target_z)
            return targets

        cfg = self.dr_config.placement
        half_xy = torch.tensor(self.tray_size_xy_np, dtype=torch.float32, device=self.device) * 0.5
        usable = torch.clamp(half_xy - float(cfg.target_xy_margin), min=0.005)
        min_sep = float(cfg.min_target_separation)
        for env_i in range(self.num_envs):
            placed_xy: list[torch.Tensor] = []
            for obj_i in range(self.object_count_max):
                best_xy = tray_center[:2]
                for _ in range(50):
                    xy = tray_center[:2] + (torch.rand((2,), device=self.device) * 2.0 - 1.0) * usable
                    if all(float(torch.linalg.norm(xy - prev).item()) >= min_sep for prev in placed_xy):
                        best_xy = xy
                        break
                placed_xy.append(best_xy)
                targets[env_i, obj_i, :2] = best_xy
                targets[env_i, obj_i, 2] = float(cfg.target_z)
        return targets

    def _apply_scripted_grasp_assist(self) -> None:
        """Carry near grasped objects for scripted demonstration collection."""
        if not self.scripted_grasp_assist or self._grasp_assist_held_idx is None:
            return
        tcp_pos = self.agent.tcp.pose.p
        obj_pos = self._get_object_pos_tensor()
        qpos = self.agent.robot.get_qpos()
        gripper_closed = torch.mean(torch.abs(qpos[:, -2:]), dim=1) < 0.015
        gripper_open = torch.mean(torch.abs(qpos[:, -2:]), dim=1) > 0.025

        valid = self._active_object_mask & (~self._placed_mask)
        dist = torch.linalg.norm(obj_pos - tcp_pos[:, None, :], dim=-1)
        dist = torch.where(valid, dist, torch.full_like(dist, 1e6))
        nearest_dist, nearest_idx = torch.min(dist, dim=1)
        can_grasp = gripper_closed & (self._grasp_assist_held_idx < 0) & (nearest_dist < self.grasp_assist_radius)
        self._grasp_assist_held_idx = torch.where(can_grasp, nearest_idx, self._grasp_assist_held_idx)
        self._grasp_assist_held_idx = torch.where(gripper_open, torch.full_like(self._grasp_assist_held_idx, -1), self._grasp_assist_held_idx)

        carry_offset = torch.tensor([0.0, 0.0, self.grasp_assist_z_offset], dtype=torch.float32, device=self.device)
        target_pos = tcp_pos + carry_offset.view(1, 3)
        for obj_idx, actor in enumerate(self.objects):
            held = self._grasp_assist_held_idx == obj_idx
            if not torch.any(held):
                continue
            p = actor.pose.p.clone()
            q = actor.pose.q.clone()
            p[held] = target_pos[held]
            actor.set_pose(Pose.create_from_pq(p=p, q=q))
            try:
                actor.set_linear_velocity(torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device))
                actor.set_angular_velocity(torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device))
            except Exception:
                pass

    def _update_target_indices(self, tcp_pos: torch.Tensor, obj_pos: torch.Tensor):
        d = torch.linalg.norm(obj_pos - tcp_pos[:, None, :], dim=-1)
        huge = torch.full_like(d, 1e6)
        valid = self._active_object_mask & (~self._placed_mask)
        d = torch.where(valid, d, huge)
        self._target_idx = torch.argmin(d, dim=1)

    def _update_placement_mask(self, info: dict[str, torch.Tensor]):
        obj_pos = self._get_object_pos_tensor()
        tray_center = torch.tensor(self.tray_center_np, dtype=torch.float32, device=self.device).view(1, 1, 3)
        tray_half = torch.tensor(self.tray_size_xy_np, dtype=torch.float32, device=self.device).view(1, 1, 2) * 0.5

        in_xy = torch.logical_and(
            torch.abs(obj_pos[..., 0] - tray_center[..., 0]) <= tray_half[..., 0],
            torch.abs(obj_pos[..., 1] - tray_center[..., 1]) <= tray_half[..., 1],
        )
        in_z = torch.logical_and(
            obj_pos[..., 2] >= self.tray_place_min_z,
            obj_pos[..., 2] <= self.tray_place_max_z,
        )

        static = []
        grasped = []
        for actor in self.objects:
            static.append(actor.is_static(lin_thresh=self.settle_speed_threshold, ang_thresh=0.5))
            grasped.append(self.agent.is_grasping(actor))
        static_mask = torch.stack(static, dim=1)
        grasped_mask = torch.stack(grasped, dim=1)

        # A tray placement should be accepted once the object is released and
        # remains inside the tray volume. Cylinders and spheres can roll or sit
        # on another object after a valid drop, so requiring a narrow surface-z
        # band makes the oracle retarget already-placed objects and pick them
        # again.
        candidate = self._active_object_mask & in_xy & in_z & (~grasped_mask)
        self._settle_counter = torch.where(candidate, self._settle_counter + 1, torch.zeros_like(self._settle_counter))
        newly_settled = self._settle_counter >= self.settle_steps_required
        self._placed_mask = self._placed_mask | newly_settled

        placed_count = torch.sum(self._placed_mask & self._active_object_mask, dim=1).float()
        total_count = torch.sum(self._active_object_mask, dim=1).float().clamp(min=1.0)
        success_fraction = placed_count / total_count
        self._last_success_fraction = success_fraction

        info["placed_count"] = placed_count
        info["total_objects"] = total_count
        info["success_fraction"] = success_fraction
        info["placed_mask"] = self._placed_mask
        info["grasped_mask"] = grasped_mask
        info["obj_pos"] = obj_pos
        info["place_targets"] = self._placement_targets

    def evaluate(self):
        info = {}
        obj_pos = self._get_object_pos_tensor()
        tcp_pos = self.agent.tcp.pose.p
        self._update_target_indices(tcp_pos, obj_pos)
        self._update_placement_mask(info)
        info["success"] = info["success_fraction"] >= (1.0 - 1e-6)
        return info

    def _compute_reward_terms(self, action: torch.Tensor, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        success_fraction = info["success_fraction"]
        progress_delta = torch.clamp(success_fraction - self._prev_success_fraction, min=0.0)
        reward_progress = self.weights.progress * progress_delta

        newly_placed = self._placed_mask & (~self._placed_counted_mask)
        place_count = torch.sum(newly_placed, dim=1).float()
        reward_place = self.weights.place_bonus * place_count
        self._placed_counted_mask = self._placed_counted_mask | newly_placed

        reward_terminal = torch.zeros_like(reward_progress)
        reward_terminal = torch.where(info["success"], reward_terminal + self.weights.terminal_success, reward_terminal)
        timed_out = (self._elapsed_steps >= self.timeout_steps) & (~info["success"])
        reward_terminal = torch.where(timed_out, reward_terminal - self.weights.terminal_timeout, reward_terminal)

        # Shaping
        obj_pos = info["obj_pos"]
        tcp_pos = self.agent.tcp.pose.p
        target_idx = self._target_idx.clamp(min=0)
        target_pos = obj_pos[torch.arange(self.num_envs, device=self.device), target_idx]

        eef_to_obj_dist = torch.linalg.norm(tcp_pos - target_pos, dim=1)
        reward_shaping = self.weights.shaping_eef_to_obj * (1.0 - torch.tanh(5 * eef_to_obj_dist))

        tray_xy = torch.tensor(self.tray_center_np[:2], dtype=torch.float32, device=self.device).view(1, 2)
        obj_to_tray = torch.linalg.norm(target_pos[:, :2] - tray_xy, dim=1)
        reward_shaping = reward_shaping + self.weights.shaping_obj_to_tray * (1.0 - torch.tanh(4 * obj_to_tray))

        grasped_target = info["grasped_mask"][torch.arange(self.num_envs, device=self.device), target_idx]
        lifted = target_pos[:, 2] > (0.022 + self.lift_height_threshold)
        reward_shaping = reward_shaping + self.weights.shaping_lift * (grasped_target & lifted).float()

        # Penalties
        penalty_drop = torch.zeros_like(reward_progress)
        if self._last_held_mask is not None:
            dropped = self._last_held_mask & (~info["grasped_mask"])
            dropped &= (~self._placed_mask)
            penalty_drop = self.weights.penalty_drop * torch.any(dropped, dim=1).float()

        penalty_collision = self.weights.penalty_collision * (tcp_pos[:, 2] < 0.02).float()

        out_workspace = torch.logical_or(
            obj_pos[..., 0] < -0.25,
            torch.logical_or(obj_pos[..., 0] > 0.35, torch.abs(obj_pos[..., 1]) > 0.45),
        )
        penalty_out = self.weights.penalty_out_of_workspace * torch.any(out_workspace & self._active_object_mask, dim=1).float()

        reward_penalties = penalty_drop + penalty_collision + penalty_out
        reward_step = torch.full_like(reward_progress, self.weights.step_cost)

        self._last_held_mask = info["grasped_mask"].clone()

        return {
            "reward_progress": reward_progress,
            "reward_place": reward_place,
            "reward_terminal": reward_terminal,
            "reward_shaping": torch.clamp(reward_shaping, min=-2.0, max=2.0),
            "reward_penalties": torch.clamp(reward_penalties, min=0.0, max=4.0),
            "reward_step": reward_step,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        terms = self._compute_reward_terms(action=action, info=info)
        self._last_reward_terms = terms
        reward = (
            terms["reward_progress"]
            + terms["reward_place"]
            + terms["reward_terminal"]
            + terms["reward_shaping"]
            - terms["reward_penalties"]
            - terms["reward_step"]
        )
        self._prev_success_fraction = info["success_fraction"].clone()
        return reward

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["success_fraction"]

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 20.0

    def _get_obs_extra(self, info: dict):
        obj_pos = self._get_object_pos_tensor().reshape(self.num_envs, -1)
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            observation_state_object=obj_pos,
            observation_state_object_mask=self._active_object_mask.float(),
            observation_state_placed_mask=self._placed_mask.float(),
            observation_state_place_targets=self._placement_targets.reshape(self.num_envs, -1),
            observation_state_task_progress=info["success_fraction"].unsqueeze(-1),
        )
        return obs

    def step(self, action):
        # Override to inject reward term breakdown into info for logging/collection.
        action = self._step_action(action)
        self._elapsed_steps += 1
        self._apply_scripted_grasp_assist()
        info = self.get_info()
        obs = self.get_obs(info, unflattened=True)
        reward = self.get_reward(obs=obs, action=action, info=info)

        if self._last_reward_terms is not None:
            info.update(self._last_reward_terms)
            info["reward_total"] = reward

        obs = self._flatten_raw_obs(obs)
        if "success" in info:
            terminated = info["success"].clone()
        else:
            terminated = torch.zeros(self.num_envs, dtype=bool, device=self.device)

        self._last_obs = obs
        return (
            obs,
            reward,
            terminated,
            torch.zeros(self.num_envs, dtype=bool, device=self.device),
            info,
        )
