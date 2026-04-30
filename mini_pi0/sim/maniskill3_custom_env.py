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
from mani_skill.utils.structs import Pose


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
        reset_settle_steps: int = 18,
        tray_z_tol: float = 0.03,
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
        **kwargs,
    ):
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
        self.reset_settle_steps = int(max(0, reset_settle_steps))
        self.tray_z_tol = float(tray_z_tol)
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
        self._target_idx = None
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
        pose = sapien_utils.look_at(eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos)
        return [CameraConfig("base_camera", pose, self.sensor_width, self.sensor_height, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=self.human_cam_eye_pos, target=self.human_cam_target_pos)
        return CameraConfig("render_camera", pose, self.render_width, self.render_height, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        self.objects: list[Any] = []
        self.object_shape_names: list[str] = []

        for idx in range(self.object_count_max):
            shape = self.object_types[idx % len(self.object_types)]
            name = f"obj_{idx:02d}_{shape}"
            if shape == "cube":
                actor = actors.build_cube(
                    self.scene,
                    half_size=0.02,
                    color=[0.90, 0.20, 0.20, 1.0],
                    name=name,
                    initial_pose=sapien.Pose(p=[5, 5, 5]),
                )
            elif shape == "sphere":
                actor = actors.build_sphere(
                    self.scene,
                    radius=0.022,
                    color=[0.18, 0.56, 0.88, 1.0],
                    name=name,
                    initial_pose=sapien.Pose(p=[5, 5, 5]),
                )
            else:
                actor = actors.build_cylinder(
                    self.scene,
                    radius=0.020,
                    half_length=0.020,
                    color=[0.85, 0.74, 0.21, 1.0],
                    name=name,
                    initial_pose=sapien.Pose(p=[5, 5, 5]),
                )
            self.objects.append(actor)
            self.object_shape_names.append(shape)

        tray_base_half_z = 0.006
        sx = float(self.tray_size_xy_np[0]) * 0.5
        sy = float(self.tray_size_xy_np[1]) * 0.5
        wt = self.tray_wall_thickness
        wh = self.tray_wall_height
        wall_half_h = wh * 0.5
        base_z = float(self.tray_center_np[2])

        self.tray = actors.build_box(
            self.scene,
            half_sizes=[sx, sy, tray_base_half_z],
            color=[0.25, 0.25, 0.28, 1.0],
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
            wall = actors.build_box(
                self.scene,
                half_sizes=hs,
                color=[0.18, 0.18, 0.20, 1.0],
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
        bowl_builder.add_nonconvex_collision_from_file(
            filename=os.path.join(bowl_assets_dir, "frl_apartment_bowl_07.ply"),
            pose=sapien.Pose(q=bowl_fix_q),
            scale=[1.0, 1.0, 1.0],
        )
        bowl_builder.add_visual_from_file(
            filename=os.path.join(bowl_assets_dir, "frl_apartment_bowl_07.glb"),
            scale=[1.0, 1.0, 1.0],
            pose=sapien.Pose(q=bowl_fix_q),
        )
        bowl_pose_p = self.bowl_center_np.copy()
        bowl_pose_p[2] += self.bowl_table_z_offset
        bowl_builder.initial_pose = sapien.Pose(p=bowl_pose_p.tolist(), q=self.bowl_pose_quat.tolist())
        self.bowl = bowl_builder.build_kinematic(name="source_bowl")

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
            self._target_idx = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
            self._last_reward_terms = None

            active_counts = torch.randint(
                low=self.object_count_min,
                high=self.object_count_max + 1,
                size=(self.num_envs,),
                device=self.device,
            )
            for i in range(self.object_count_max):
                self._active_object_mask[:, i] = i < active_counts

            # Spawn active objects inside bowl with simple separation retries.
            device = self.device
            bowl_center_xy = torch.tensor(self.bowl_center_np[:2], dtype=torch.float32, device=device)
            spawn_r = max(0.02, self.bowl_inner_radius - 0.02)
            min_sep = 0.045
            xyz_store: list[torch.Tensor] = []
            for i, actor in enumerate(self.objects):
                xy = torch.zeros((self.num_envs, 2), device=device)
                for env_i in range(self.num_envs):
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
            if self.reset_settle_steps > 0:
                for _ in range(self.reset_settle_steps):
                    self.scene.step()
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
        in_z = torch.abs(obj_pos[..., 2] - 0.022) <= self.tray_z_tol

        static = []
        grasped = []
        for actor in self.objects:
            static.append(actor.is_static(lin_thresh=self.settle_speed_threshold, ang_thresh=0.5))
            grasped.append(self.agent.is_grasping(actor))
        static_mask = torch.stack(static, dim=1)
        grasped_mask = torch.stack(grasped, dim=1)

        candidate = self._active_object_mask & in_xy & in_z & static_mask & (~grasped_mask)
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
            observation_state_task_progress=info["success_fraction"].unsqueeze(-1),
        )
        return obs

    def step(self, action):
        # Override to inject reward term breakdown into info for logging/collection.
        action = self._step_action(action)
        self._elapsed_steps += 1
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
