"""Training dataset construction for robomimic HDF5 and LeRobot v3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mini_pi0.config.io import dump_config
from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys
from mini_pi0.dataset.episodes import load_episodes_from_config
from mini_pi0.dataset.lerobot_v3 import (
    LEROBOT_V3_FORMATS,
    LeRobotActionStatsComputer,
    LeRobotDatasetFactory,
    LeRobotFeatureSpec,
    LeRobotPolicyDataset,
    LeRobotTemporalConfig,
    LeRobotV3OpenConfig,
)
from mini_pi0.dataset.stats import ActionStats
from mini_pi0.dataset.torch_dataset import ActionChunkDataset
from mini_pi0.train.data import (
    curate_episodes,
    infer_action_dim,
    infer_prop_dim,
    split_train_val,
    validate_image_observations,
)


@dataclass(frozen=True)
class PreparedTrainingData:
    """Datasets and metadata needed by the training loop."""

    episode_count: int
    train_dataset: Any
    val_dataset: Any | None
    action_stats: ActionStats
    action_stats_path: Path
    curation_summary: dict[str, Any]
    normalize_actions_on_device: bool


class TrainingDatasetBuilder:
    """Build training datasets from the configured data backend."""

    def __init__(self, cfg: RootConfig, run_dir: Path) -> None:
        """Create a builder for one training run."""

        self.cfg = cfg
        self.run_dir = run_dir

    def build(self) -> PreparedTrainingData:
        """Build and split the configured training dataset."""

        fmt = str(self.cfg.data.format).strip().lower()
        if fmt in LEROBOT_V3_FORMATS:
            return self._build_lerobot_v3()
        return self._build_robomimic_hdf5()

    def _build_robomimic_hdf5(self) -> PreparedTrainingData:
        """Build the existing in-memory robomimic dataset path."""

        print(
            "[train] Loading dataset | "
            f"format={self.cfg.data.format} n_demos={self.cfg.data.n_demos}",
            flush=True,
        )
        episodes = load_episodes_from_config(self.cfg)
        print(f"[train] Dataset loaded | episodes={len(episodes)}", flush=True)
        episodes, curation_summary = curate_episodes(episodes, self.cfg)
        self._print_curation(curation_summary)

        state_keys = effective_state_keys(self.cfg.robot)
        image_keys = effective_image_keys(self.cfg.robot)
        inferred_action_dim = infer_action_dim(episodes)
        inferred_prop_dim = infer_prop_dim(episodes[0].obs[0], state_keys)
        validate_image_observations(episodes[0].obs[0], image_keys)
        self._align_dims(inferred_action_dim, inferred_prop_dim)

        all_actions = np.concatenate([ep.actions.astype(np.float32) for ep in episodes], axis=0)
        stats = ActionStats.from_actions(all_actions)
        stats_path = self._save_stats(stats)
        dataset = ActionChunkDataset(
            episodes=episodes,
            chunk_size=self.cfg.data.chunk_size,
            image_key=self.cfg.robot.image_key,
            image_keys=image_keys,
            proprio_keys=state_keys,
            action_stats=stats,
            obs_horizon=int(getattr(self.cfg.model, "obs_horizon", 1)),
            preserve_camera_dim=self._preserve_camera_dim(),
        )
        train_dataset, val_dataset = self._split(dataset)
        print(
            f"[train] Prepared action-chunk dataset | total={len(dataset)} train={len(train_dataset)} "
            f"val={(len(val_dataset) if val_dataset is not None else 0)}",
            flush=True,
        )
        return PreparedTrainingData(
            episode_count=len(episodes),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            action_stats=stats,
            action_stats_path=stats_path,
            curation_summary=curation_summary,
            normalize_actions_on_device=False,
        )

    def _build_lerobot_v3(self) -> PreparedTrainingData:
        """Build the lazy LeRobot v3 training path."""

        repo_id = self.cfg.data.lerobot_repo_id
        if not repo_id:
            raise ValueError("data.lerobot_repo_id must be set when data.format=lerobot_v3")
        image_keys = self.cfg.data.lerobot_image_keys or effective_image_keys(self.cfg.robot)
        spec = LeRobotFeatureSpec.from_keys(
            action_key=self.cfg.data.lerobot_action_key,
            state_key=self.cfg.data.lerobot_state_key,
            image_keys=image_keys,
            episode_index_key=self.cfg.data.lerobot_episode_index_key,
        )
        open_cfg = LeRobotV3OpenConfig(
            repo_id=str(repo_id),
            root=self.cfg.data.lerobot_root,
            revision=self.cfg.data.lerobot_revision,
            episodes=self.cfg.data.lerobot_episodes,
            local_files_only=bool(self.cfg.data.lerobot_local_files_only),
            video_backend=self.cfg.data.lerobot_video_backend,
        )
        plain_dataset = LeRobotDatasetFactory(open_cfg).open()
        spec.validate(plain_dataset)
        stats = ActionStats.from_iterable(LeRobotActionStatsComputer(plain_dataset, spec.action_key).iter_actions())
        stats_path = self._save_stats(stats)

        temporal = LeRobotTemporalConfig(
            fps=int(getattr(plain_dataset, "fps", self.cfg.simulator.control_freq)),
            obs_horizon=int(getattr(self.cfg.model, "obs_horizon", 1)),
            chunk_size=int(self.cfg.data.chunk_size),
        )
        temporal_dataset = LeRobotDatasetFactory(open_cfg, temporal).open(spec)
        dataset = LeRobotPolicyDataset(
            dataset=temporal_dataset,
            spec=spec,
            chunk_size=int(self.cfg.data.chunk_size),
            obs_horizon=int(getattr(self.cfg.model, "obs_horizon", 1)),
            preserve_camera_dim=self._preserve_camera_dim(),
        )
        info = dataset.info()
        self._align_dims(info.action_dim, info.prop_dim)
        train_dataset, val_dataset = self._split(dataset)
        curation_summary = {
            "enabled": False,
            "reason": "LeRobot v3 path uses lazy indexed loading; episode curation is not applied.",
        }
        print(
            f"[train] Prepared lazy LeRobot dataset | episodes={info.episode_count} total={len(dataset)} "
            f"train={len(train_dataset)} val={(len(val_dataset) if val_dataset is not None else 0)}",
            flush=True,
        )
        return PreparedTrainingData(
            episode_count=info.episode_count,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            action_stats=stats,
            action_stats_path=stats_path,
            curation_summary=curation_summary,
            normalize_actions_on_device=True,
        )

    def _align_dims(self, action_dim: int, prop_dim: int) -> None:
        """Align config dimensions to dataset dimensions."""

        if self.cfg.robot.action_dim != action_dim:
            print(f"[train] Overriding robot.action_dim from {self.cfg.robot.action_dim} to inferred {action_dim}.")
        if self.cfg.model.action_dim != action_dim:
            print(f"[train] Overriding model.action_dim from {self.cfg.model.action_dim} to inferred {action_dim}.")
        if self.cfg.model.prop_dim != prop_dim:
            print(f"[train] Overriding model.prop_dim from {self.cfg.model.prop_dim} to inferred {prop_dim}.")
        self.cfg.robot.action_dim = action_dim
        self.cfg.model.action_dim = action_dim
        self.cfg.model.prop_dim = prop_dim
        dump_config(self.run_dir / "config_resolved.yaml", self.cfg)

    def _save_stats(self, stats: ActionStats) -> Path:
        """Persist action stats into the current run artifacts."""

        path = self.run_dir / "artifacts" / "action_stats.json"
        stats.save(str(path))
        return path

    def _split(self, dataset: Any) -> tuple[Any, Any | None]:
        """Split train/validation datasets consistently."""

        return split_train_val(
            dataset,
            val_ratio=float(getattr(self.cfg.train, "val_ratio", 0.0)),
            seed=int(self.cfg.experiment.seed),
        )

    def _preserve_camera_dim(self) -> bool:
        """Return whether cross-attention expects separate camera tokens."""

        return str(getattr(self.cfg.model, "conditioning_mode", "global")).strip().lower() == "cross_attention"

    @staticmethod
    def _print_curation(curation_summary: dict[str, Any]) -> None:
        """Print curation summary for the robomimic path."""

        if not curation_summary.get("enabled", False):
            return
        print(
            "[train] Data curation | "
            f"before={curation_summary['before_episodes']} after={curation_summary['after_episodes']} "
            f"removed={curation_summary['removed_episodes']} progress_key={curation_summary.get('progress_key')}",
            flush=True,
        )
        if curation_summary.get("reasons"):
            print(f"[train] Data curation reasons | {curation_summary['reasons']}", flush=True)
