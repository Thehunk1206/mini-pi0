from __future__ import annotations

import argparse
import json
from typing import Any

from mini_pi0.config.io import load_config
from mini_pi0.dataset.episodes import list_supported_dataset_formats
from mini_pi0.dataset.robomimic_download import download_robomimic_dataset
from mini_pi0.dataset.robot_dataset_mapping import build_robot_dataset_mapping
from mini_pi0.eval.ablation import run_eval_ablation
from mini_pi0.eval.runner import run_eval
from mini_pi0.sim.registry import backend_status, list_backends
from mini_pi0.train.runner import run_train
from mini_pi0.vision.precompute import run_precompute_vision
from mini_pi0.vision.encoders import list_timm_model_options, list_torchvision_model_options


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    """Register shared config/override flags for a subcommand parser.

    Args:
        parser: Subcommand parser instance to mutate.
    """

    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted override. Can be repeated, e.g. --set train.epochs=50",
    )


def _append_override(overrides: list[str], key: str, value: Any) -> None:
    """Append a dotted override only when value is explicitly provided.

    Args:
        overrides: Mutable list collecting ``key=value`` entries.
        key: Dotted config key path.
        value: Override value; ignored if ``None``.
    """

    if value is not None:
        overrides.append(f"{key}={value}")


def _parse_csv_values(text: str | None, cast_type):
    """Parse comma-separated CLI text values into typed list."""

    if text is None:
        return []
    out = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        out.append(cast_type(item))
    return out


def _apply_train_overrides(args: argparse.Namespace) -> list[str]:
    """Translate ``train`` CLI args into dotted config overrides.

    Args:
        args: Parsed CLI namespace for train command.

    Returns:
        Ordered list of dotted overrides.
    """

    overrides = list(args.overrides or [])

    _append_override(overrides, "experiment.name", args.run_name)
    _append_override(overrides, "experiment.seed", args.seed)

    _append_override(overrides, "simulator.backend", args.backend)
    _append_override(overrides, "simulator.task", args.task)
    _append_override(overrides, "simulator.robot", args.robot)
    _append_override(overrides, "simulator.controller", args.controller)
    _append_override(overrides, "robot.image_key", args.image_key)
    if args.image_keys is not None:
        keys = _parse_csv_values(args.image_keys, str)
        if keys:
            _append_override(overrides, "robot.image_keys", keys)
            _append_override(overrides, "robot.image_key", keys[0])

    _append_override(overrides, "data.format", args.data_format)
    _append_override(overrides, "data.observation_mode", args.observation_mode)
    _append_override(overrides, "data.robomimic_hdf5", args.robomimic_hdf5)
    _append_override(overrides, "data.robomimic_data_group", args.robomimic_data_group)
    _append_override(overrides, "data.lerobot_repo_id", args.lerobot_repo_id)
    _append_override(overrides, "data.lerobot_action_key", args.lerobot_action_key)
    _append_override(overrides, "data.lerobot_episode_index_key", args.lerobot_episode_index_key)
    _append_override(overrides, "data.lerobot_local_files_only", args.lerobot_local_files_only)
    _append_override(overrides, "data.lerobot_video_backend", args.lerobot_video_backend)
    _append_override(overrides, "data.precomputed_features_path", args.precomputed_features_path)
    _append_override(overrides, "data.precomputed_feature_key", args.precomputed_feature_key)
    if args.fallback_image_hw is not None:
        _append_override(overrides, "data.fallback_image_hw", list(args.fallback_image_hw))
    _append_override(overrides, "data.n_demos", args.n_demos)
    _append_override(overrides, "data.chunk_size", args.chunk_size)
    _append_override(overrides, "data.action_stats_path", args.action_stats)
    _append_override(overrides, "data.filter_min_episode_length", args.filter_min_episode_length)
    _append_override(overrides, "data.filter_min_action_std", args.filter_min_action_std)
    _append_override(overrides, "data.filter_min_state_delta", args.filter_min_state_delta)
    _append_override(overrides, "data.filter_state_delta_key", args.filter_state_delta_key)
    _append_override(overrides, "data.filter_drop_nan", args.filter_drop_nan)

    _append_override(overrides, "robot.action_dim", args.action_dim)
    _append_override(overrides, "model.action_dim", args.action_dim)
    _append_override(overrides, "model.obs_mode", args.model_obs_mode)
    _append_override(overrides, "model.vision_dim", args.model_vision_dim)
    _append_override(overrides, "model.chunk_size", args.chunk_size)

    _append_override(overrides, "train.epochs", args.epochs)
    _append_override(overrides, "train.batch_size", args.batch_size)
    _append_override(overrides, "train.lr", args.lr)
    _append_override(overrides, "train.lr_scheduler", args.lr_scheduler)
    _append_override(overrides, "train.scheduler_t_max", args.scheduler_t_max)
    _append_override(overrides, "train.scheduler_eta_min", args.scheduler_eta_min)
    _append_override(overrides, "train.scheduler_step_size", args.scheduler_step_size)
    _append_override(overrides, "train.scheduler_gamma", args.scheduler_gamma)
    _append_override(overrides, "train.resume_from", args.resume_from)
    _append_override(overrides, "train.resume_optimizer", args.resume_optimizer)
    _append_override(overrides, "train.val_ratio", args.val_ratio)
    _append_override(overrides, "train.ema_decay", args.ema_decay)
    _append_override(overrides, "train.checkpoint_use_ema", args.checkpoint_use_ema)
    _append_override(overrides, "train.device", args.device)
    _append_override(overrides, "train.model_print_depth", args.model_print_depth)
    _append_override(overrides, "train.num_workers", args.num_workers)
    _append_override(overrides, "train.persistent_workers", args.persistent_workers)
    _append_override(overrides, "train.save_best", args.save_best)
    _append_override(overrides, "train.save_best_min_delta", args.save_best_min_delta)

    _append_override(overrides, "vision.backend", args.vision_backend)
    _append_override(overrides, "vision.model_name", args.vision_model_name)
    _append_override(overrides, "vision.pretrained", args.vision_pretrained)
    _append_override(overrides, "vision.batch_size", args.vision_batch_size)
    _append_override(overrides, "vision.image_size", args.vision_image_size)
    _append_override(overrides, "vision.output_path", args.vision_output_path)
    _append_override(overrides, "vision.use_runtime_extractor", args.vision_use_runtime_extractor)
    _append_override(overrides, "vision.hf_model_id", args.vision_hf_model_id)
    _append_override(overrides, "vision.local_files_only", args.vision_local_files_only)

    return overrides


def _apply_precompute_overrides(args: argparse.Namespace) -> list[str]:
    """Translate ``precompute-vision`` CLI args into dotted config overrides."""

    overrides = list(args.overrides or [])
    _append_override(overrides, "experiment.name", args.run_name)
    _append_override(overrides, "experiment.seed", args.seed)

    _append_override(overrides, "data.format", args.data_format)
    _append_override(overrides, "data.robomimic_hdf5", args.robomimic_hdf5)
    _append_override(overrides, "data.robomimic_data_group", args.robomimic_data_group)
    _append_override(overrides, "data.lerobot_repo_id", args.lerobot_repo_id)
    _append_override(overrides, "data.lerobot_action_key", args.lerobot_action_key)
    _append_override(overrides, "data.lerobot_episode_index_key", args.lerobot_episode_index_key)
    _append_override(overrides, "data.lerobot_local_files_only", args.lerobot_local_files_only)
    _append_override(overrides, "data.lerobot_video_backend", args.lerobot_video_backend)
    _append_override(overrides, "data.n_demos", args.n_demos)
    _append_override(overrides, "data.precomputed_features_path", args.precomputed_features_path)
    _append_override(overrides, "data.precomputed_feature_key", args.precomputed_feature_key)

    _append_override(overrides, "robot.image_key", args.image_key)
    if args.image_keys is not None:
        keys = _parse_csv_values(args.image_keys, str)
        if keys:
            _append_override(overrides, "robot.image_keys", keys)
            _append_override(overrides, "robot.image_key", keys[0])

    _append_override(overrides, "vision.backend", args.vision_backend)
    _append_override(overrides, "vision.model_name", args.vision_model_name)
    _append_override(overrides, "vision.pretrained", args.vision_pretrained)
    _append_override(overrides, "vision.batch_size", args.vision_batch_size)
    _append_override(overrides, "vision.image_size", args.vision_image_size)
    _append_override(overrides, "vision.output_path", args.vision_output_path)
    _append_override(overrides, "vision.hf_model_id", args.vision_hf_model_id)
    _append_override(overrides, "vision.local_files_only", args.vision_local_files_only)
    return overrides


def _apply_eval_overrides(args: argparse.Namespace) -> list[str]:
    """Translate ``eval`` CLI args into dotted config overrides.

    Args:
        args: Parsed CLI namespace for eval command.

    Returns:
        Ordered list of dotted overrides.
    """

    overrides = list(args.overrides or [])

    _append_override(overrides, "experiment.name", args.run_name)
    _append_override(overrides, "experiment.seed", args.seed)

    _append_override(overrides, "simulator.backend", args.backend)
    _append_override(overrides, "simulator.task", args.task)
    _append_override(overrides, "simulator.robot", args.robot)
    _append_override(overrides, "simulator.controller", args.controller)
    _append_override(overrides, "robot.image_key", args.image_key)
    if args.image_keys is not None:
        keys = _parse_csv_values(args.image_keys, str)
        if keys:
            _append_override(overrides, "robot.image_keys", keys)
            _append_override(overrides, "robot.image_key", keys[0])

    _append_override(overrides, "eval.checkpoint", args.checkpoint)
    _append_override(overrides, "eval.run_dir", args.eval_run_dir)
    _append_override(overrides, "eval.action_stats_path", args.action_stats)
    _append_override(overrides, "eval.n_episodes", args.n_episodes)
    _append_override(overrides, "eval.execute_steps", args.execute_steps)
    _append_override(overrides, "eval.n_flow_steps", args.n_flow_steps)
    _append_override(overrides, "eval.max_steps", args.max_steps)
    _append_override(overrides, "eval.strict_parity", args.strict_parity)
    _append_override(overrides, "eval.verbose", args.verbose)
    _append_override(overrides, "eval.log_every_episodes", args.log_every_episodes)
    _append_override(overrides, "eval.action_smoothing_alpha", args.action_smoothing_alpha)
    _append_override(overrides, "eval.failure_reward_threshold", args.failure_reward_threshold)
    _append_override(overrides, "eval.device", args.device)
    _append_override(overrides, "eval.record", args.record)
    _append_override(overrides, "eval.record_grid", args.record_grid)
    _append_override(overrides, "eval.grid_size", args.grid_size)
    _append_override(overrides, "eval.grid_fps", args.grid_fps)
    _append_override(overrides, "eval.grid_width", args.grid_width)
    _append_override(overrides, "eval.grid_height", args.grid_height)
    _append_override(overrides, "eval.plot_path", args.plot_path)
    _append_override(overrides, "eval.stability_warmup_steps", args.stability_warmup_steps)
    _append_override(overrides, "eval.stability_warmup_execute_steps", args.stability_warmup_execute_steps)
    _append_override(overrides, "eval.stability_warmup_n_flow_steps", args.stability_warmup_n_flow_steps)
    _append_override(
        overrides,
        "eval.stability_warmup_action_smoothing_alpha",
        args.stability_warmup_action_smoothing_alpha,
    )
    if args.action_scale is not None:
        _append_override(overrides, "eval.action_scale", list(args.action_scale))

    if args.cube_xy is not None:
        _append_override(overrides, "eval.cube_xy", list(args.cube_xy))
    if args.cube_xy_range is not None:
        _append_override(overrides, "eval.cube_xy_range", list(args.cube_xy_range))
    _append_override(overrides, "eval.cube_z", args.cube_z)
    _append_override(overrides, "eval.cube_yaw_deg", args.cube_yaw_deg)

    return overrides


def _apply_deploy_sim_overrides(args: argparse.Namespace) -> list[str]:
    """Translate ``deploy-sim`` CLI args into dotted config overrides.

    Args:
        args: Parsed CLI namespace for deploy-sim command.

    Returns:
        Ordered list of dotted overrides.
    """

    overrides = list(args.overrides or [])

    _append_override(overrides, "experiment.name", args.run_name)
    _append_override(overrides, "experiment.seed", args.seed)

    _append_override(overrides, "simulator.backend", args.backend)
    _append_override(overrides, "simulator.task", args.task)
    _append_override(overrides, "simulator.robot", args.robot)
    _append_override(overrides, "simulator.controller", args.controller)
    _append_override(overrides, "robot.image_key", args.image_key)
    if args.image_keys is not None:
        keys = _parse_csv_values(args.image_keys, str)
        if keys:
            _append_override(overrides, "robot.image_keys", keys)
            _append_override(overrides, "robot.image_key", keys[0])

    _append_override(overrides, "deploy.mode", "sim")
    _append_override(overrides, "deploy.checkpoint", args.checkpoint)
    _append_override(overrides, "deploy.action_stats_path", args.action_stats)
    _append_override(overrides, "deploy.execute_steps", args.execute_steps)
    _append_override(overrides, "deploy.n_flow_steps", args.n_flow_steps)
    _append_override(overrides, "deploy.max_steps", args.max_steps)
    _append_override(overrides, "deploy.strict_parity", args.strict_parity)
    _append_override(overrides, "deploy.action_smoothing_alpha", args.action_smoothing_alpha)
    _append_override(overrides, "deploy.stability_warmup_steps", args.stability_warmup_steps)
    _append_override(overrides, "deploy.stability_warmup_execute_steps", args.stability_warmup_execute_steps)
    _append_override(overrides, "deploy.stability_warmup_n_flow_steps", args.stability_warmup_n_flow_steps)
    _append_override(
        overrides,
        "deploy.stability_warmup_action_smoothing_alpha",
        args.stability_warmup_action_smoothing_alpha,
    )
    _append_override(overrides, "deploy.device", args.device)
    _append_override(overrides, "deploy.record_path", args.record_path)
    if args.action_scale is not None:
        _append_override(overrides, "deploy.action_scale", list(args.action_scale))

    return overrides


def _build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser with all subcommands.

    Returns:
        Fully configured ``argparse`` parser.
    """

    p = argparse.ArgumentParser(
        description="mini-pi0 modular CLI (multi-sim training / eval / deploy)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_backends = sub.add_parser("backends", help="List simulator backend availability diagnostics")
    _add_common_config_args(p_backends)

    p_download = sub.add_parser("download-robomimic", help="Download robomimic v0.1 dataset HDF5")
    p_download.add_argument("--task", choices=["lift", "can", "square", "transport", "tool_hang"], default="lift")
    p_download.add_argument("--dataset_type", choices=["ph", "mh", "mg"], default="ph")
    p_download.add_argument("--hdf5_type", choices=["low_dim", "low_dim_sparse", "low_dim_dense"], default="low_dim")
    p_download.add_argument("--download_dir", default="data/robomimic")
    p_download.add_argument("--version", default="v1.5")
    p_download.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)

    p_map = sub.add_parser(
        "robot-dataset-map",
        help="List local robosuite robots and equivalent Hugging Face dataset mappings",
    )
    p_map.add_argument("--version", default="v1.5", help="robomimic dataset version segment")
    p_map.add_argument("--include_lerobot", action=argparse.BooleanOptionalAction, default=True)

    p_vmodels = sub.add_parser("vision-models", help="List selectable vision backbone model names")
    p_vmodels.add_argument("--backend", choices=["torchvision", "timm", "all"], default="all")
    p_vmodels.add_argument(
        "--all_timm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="List full timm model registry (can be long).",
    )

    p_precomp = sub.add_parser("precompute-vision", help="Precompute vision features for fast policy training")
    _add_common_config_args(p_precomp)
    p_precomp.add_argument("--run_name", default=None)
    p_precomp.add_argument("--seed", type=int, default=None)
    p_precomp.add_argument("--data_format", choices=list_supported_dataset_formats(), default=None)
    p_precomp.add_argument("--robomimic_hdf5", default=None)
    p_precomp.add_argument("--robomimic_data_group", default=None)
    p_precomp.add_argument("--lerobot_repo_id", default=None)
    p_precomp.add_argument("--lerobot_action_key", default=None)
    p_precomp.add_argument("--lerobot_episode_index_key", default=None)
    p_precomp.add_argument("--lerobot_local_files_only", action=argparse.BooleanOptionalAction, default=None)
    p_precomp.add_argument(
        "--lerobot_video_backend",
        choices=["pyav", "torchcodec", "video_reader"],
        default=None,
        help="LeRobot video decoder backend. Use pyav on macOS.",
    )
    p_precomp.add_argument("--n_demos", type=int, default=None)
    p_precomp.add_argument("--image_key", default=None)
    p_precomp.add_argument("--image_keys", default=None, help="Comma-separated image observation keys.")
    p_precomp.add_argument("--precomputed_features_path", default=None)
    p_precomp.add_argument("--precomputed_feature_key", default=None)
    p_precomp.add_argument("--vision_backend", choices=["torchvision", "timm", "hf"], default=None)
    p_precomp.add_argument("--vision_model_name", default=None)
    p_precomp.add_argument("--vision_pretrained", action=argparse.BooleanOptionalAction, default=None)
    p_precomp.add_argument("--vision_batch_size", type=int, default=None)
    p_precomp.add_argument("--vision_image_size", type=int, default=None)
    p_precomp.add_argument("--vision_output_path", default=None)
    p_precomp.add_argument("--vision_hf_model_id", default=None)
    p_precomp.add_argument("--vision_local_files_only", action=argparse.BooleanOptionalAction, default=None)

    p_train = sub.add_parser("train", help="Train action model")
    _add_common_config_args(p_train)
    p_train.add_argument("--run_name", default=None)
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument("--backend", choices=list_backends(), default=None)
    p_train.add_argument("--task", default=None)
    p_train.add_argument("--robot", default=None)
    p_train.add_argument("--controller", default=None)
    p_train.add_argument("--image_key", default=None)
    p_train.add_argument("--image_keys", default=None, help="Comma-separated image observation keys.")
    p_train.add_argument("--data_format", choices=list_supported_dataset_formats(), default=None)
    p_train.add_argument("--observation_mode", choices=["image", "precomputed"], default=None)
    p_train.add_argument("--robomimic_hdf5", default=None)
    p_train.add_argument("--robomimic_data_group", default=None)
    p_train.add_argument("--lerobot_repo_id", default=None)
    p_train.add_argument("--lerobot_action_key", default=None)
    p_train.add_argument("--lerobot_episode_index_key", default=None)
    p_train.add_argument("--lerobot_local_files_only", action=argparse.BooleanOptionalAction, default=None)
    p_train.add_argument(
        "--lerobot_video_backend",
        choices=["pyav", "torchcodec", "video_reader"],
        default=None,
        help="LeRobot video decoder backend. Use pyav on macOS.",
    )
    p_train.add_argument("--precomputed_features_path", default=None)
    p_train.add_argument("--precomputed_feature_key", default=None)
    p_train.add_argument("--fallback_image_hw", type=int, nargs=2, default=None)
    p_train.add_argument("--n_demos", type=int, default=None)
    p_train.add_argument("--chunk_size", type=int, default=None)
    p_train.add_argument("--filter_min_episode_length", type=int, default=None)
    p_train.add_argument("--filter_min_action_std", type=float, default=None)
    p_train.add_argument("--filter_min_state_delta", type=float, default=None)
    p_train.add_argument("--filter_state_delta_key", default=None)
    p_train.add_argument("--filter_drop_nan", action=argparse.BooleanOptionalAction, default=None)
    p_train.add_argument("--action_dim", type=int, default=None)
    p_train.add_argument("--model_obs_mode", choices=["image", "feature"], default=None)
    p_train.add_argument("--model_vision_dim", type=int, default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch_size", type=int, default=None)
    p_train.add_argument("--lr", type=float, default=None)
    p_train.add_argument("--lr_scheduler", choices=["cosine", "step", "none"], default=None)
    p_train.add_argument("--scheduler_t_max", type=int, default=None)
    p_train.add_argument("--scheduler_eta_min", type=float, default=None)
    p_train.add_argument("--scheduler_step_size", type=int, default=None)
    p_train.add_argument("--scheduler_gamma", type=float, default=None)
    p_train.add_argument("--resume_from", default=None)
    p_train.add_argument("--resume_optimizer", action=argparse.BooleanOptionalAction, default=None)
    p_train.add_argument("--val_ratio", type=float, default=None)
    p_train.add_argument("--ema_decay", type=float, default=None)
    p_train.add_argument("--checkpoint_use_ema", action=argparse.BooleanOptionalAction, default=None)
    p_train.add_argument("--action_stats", default=None)
    p_train.add_argument("--device", default=None)
    p_train.add_argument("--model_print_depth", type=int, default=None)
    p_train.add_argument("--num_workers", type=int, default=None)
    p_train.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=None)
    p_train.add_argument("--save_best", action=argparse.BooleanOptionalAction, default=None)
    p_train.add_argument("--save_best_min_delta", type=float, default=None)
    p_train.add_argument("--vision_backend", choices=["torchvision", "timm", "hf"], default=None)
    p_train.add_argument("--vision_model_name", default=None)
    p_train.add_argument("--vision_pretrained", action=argparse.BooleanOptionalAction, default=None)
    p_train.add_argument("--vision_batch_size", type=int, default=None)
    p_train.add_argument("--vision_image_size", type=int, default=None)
    p_train.add_argument("--vision_output_path", default=None)
    p_train.add_argument("--vision_use_runtime_extractor", action=argparse.BooleanOptionalAction, default=None)
    p_train.add_argument("--vision_hf_model_id", default=None)
    p_train.add_argument("--vision_local_files_only", action=argparse.BooleanOptionalAction, default=None)

    p_eval = sub.add_parser("eval", help="Evaluate model in simulation")
    _add_common_config_args(p_eval)
    p_eval.add_argument("--run_name", default=None)
    p_eval.add_argument("--seed", type=int, default=None)
    p_eval.add_argument("--backend", choices=list_backends(), default=None)
    p_eval.add_argument("--task", default=None)
    p_eval.add_argument("--robot", default=None)
    p_eval.add_argument("--controller", default=None)
    p_eval.add_argument("--image_key", default=None)
    p_eval.add_argument("--image_keys", default=None, help="Comma-separated image observation keys.")
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument("--eval_run_dir", default=None)
    p_eval.add_argument("--action_stats", default=None)
    p_eval.add_argument("--n_episodes", type=int, default=None)
    p_eval.add_argument("--execute_steps", type=int, default=None)
    p_eval.add_argument("--n_flow_steps", type=int, default=None)
    p_eval.add_argument("--max_steps", type=int, default=None)
    p_eval.add_argument("--strict_parity", action=argparse.BooleanOptionalAction, default=None)
    p_eval.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None)
    p_eval.add_argument("--log_every_episodes", type=int, default=None)
    p_eval.add_argument("--action_smoothing_alpha", type=float, default=None)
    p_eval.add_argument("--action_scale", type=float, nargs="+", default=None)
    p_eval.add_argument("--failure_reward_threshold", type=float, default=None)
    p_eval.add_argument("--device", default=None)
    p_eval.add_argument("--record", action=argparse.BooleanOptionalAction, default=None)
    p_eval.add_argument("--record_grid", action=argparse.BooleanOptionalAction, default=None)
    p_eval.add_argument("--grid_size", type=int, default=None)
    p_eval.add_argument("--grid_fps", type=int, default=None)
    p_eval.add_argument("--grid_width", type=int, default=None)
    p_eval.add_argument("--grid_height", type=int, default=None)
    p_eval.add_argument("--plot_path", default=None)
    p_eval.add_argument("--stability_warmup_steps", type=int, default=None)
    p_eval.add_argument("--stability_warmup_execute_steps", type=int, default=None)
    p_eval.add_argument("--stability_warmup_n_flow_steps", type=int, default=None)
    p_eval.add_argument("--stability_warmup_action_smoothing_alpha", type=float, default=None)
    p_eval.add_argument("--cube_xy", type=float, nargs=2, default=None)
    p_eval.add_argument("--cube_xy_range", type=float, nargs=4, default=None)
    p_eval.add_argument("--cube_z", type=float, default=None)
    p_eval.add_argument("--cube_yaw_deg", type=float, default=None)

    p_ablate = sub.add_parser("ablate-eval", help="Run eval ablations over rollout hyperparameters")
    _add_common_config_args(p_ablate)
    p_ablate.add_argument("--run_name", default=None)
    p_ablate.add_argument("--seed", type=int, default=None)
    p_ablate.add_argument("--checkpoint", default=None)
    p_ablate.add_argument("--action_stats", default=None)
    p_ablate.add_argument("--n_episodes", type=int, default=None)
    p_ablate.add_argument("--max_steps", type=int, default=None)
    p_ablate.add_argument("--strict_parity", action=argparse.BooleanOptionalAction, default=None)
    p_ablate.add_argument(
        "--execute_steps_values",
        default="1,2,4,8",
        help="Comma-separated execute_steps values for ablation.",
    )
    p_ablate.add_argument(
        "--n_flow_steps_values",
        default="10,15,30",
        help="Comma-separated n_flow_steps values for ablation.",
    )
    p_ablate.add_argument(
        "--smoothing_values",
        default="0.0,0.2,0.4",
        help="Comma-separated action_smoothing_alpha values for ablation.",
    )

    p_deploy = sub.add_parser("deploy-sim", help="Run simulation deployment loop")
    _add_common_config_args(p_deploy)
    p_deploy.add_argument("--run_name", default=None)
    p_deploy.add_argument("--seed", type=int, default=None)
    p_deploy.add_argument("--backend", choices=list_backends(), default=None)
    p_deploy.add_argument("--task", default=None)
    p_deploy.add_argument("--robot", default=None)
    p_deploy.add_argument("--controller", default=None)
    p_deploy.add_argument("--image_key", default=None)
    p_deploy.add_argument("--image_keys", default=None, help="Comma-separated image observation keys.")
    p_deploy.add_argument("--checkpoint", default=None)
    p_deploy.add_argument("--action_stats", default=None)
    p_deploy.add_argument("--execute_steps", type=int, default=None)
    p_deploy.add_argument("--n_flow_steps", type=int, default=None)
    p_deploy.add_argument("--max_steps", type=int, default=None)
    p_deploy.add_argument("--strict_parity", action=argparse.BooleanOptionalAction, default=None)
    p_deploy.add_argument("--action_smoothing_alpha", type=float, default=None)
    p_deploy.add_argument("--stability_warmup_steps", type=int, default=None)
    p_deploy.add_argument("--stability_warmup_execute_steps", type=int, default=None)
    p_deploy.add_argument("--stability_warmup_n_flow_steps", type=int, default=None)
    p_deploy.add_argument("--stability_warmup_action_smoothing_alpha", type=float, default=None)
    p_deploy.add_argument("--action_scale", type=float, nargs="+", default=None)
    p_deploy.add_argument("--device", default=None)
    p_deploy.add_argument("--record_path", default=None)

    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``mini-pi0`` commands.

    Args:
        argv: Optional argv override for tests/programmatic calls.

    Returns:
        Process-style exit code.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "backends":
        print(json.dumps(backend_status(), indent=2, sort_keys=True))
        return 0

    if args.command == "download-robomimic":
        out = download_robomimic_dataset(
            task=args.task,
            dataset_type=args.dataset_type,
            hdf5_type=args.hdf5_type,
            download_dir=args.download_dir,
            version=args.version,
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    if args.command == "robot-dataset-map":
        out = build_robot_dataset_mapping(version=args.version, include_lerobot=bool(args.include_lerobot))
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    if args.command == "vision-models":
        out: dict[str, Any] = {}
        if args.backend in {"torchvision", "all"}:
            out["torchvision"] = list_torchvision_model_options()
        if args.backend in {"timm", "all"}:
            out["timm"] = list_timm_model_options(only_recommended=not bool(args.all_timm))
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    if args.command == "precompute-vision":
        cfg = load_config(args.config, overrides=_apply_precompute_overrides(args))
        run_precompute_vision(cfg)
        return 0

    if args.command == "train":
        cfg = load_config(args.config, overrides=_apply_train_overrides(args))
        run_train(cfg)
        return 0

    if args.command == "eval":
        cfg = load_config(args.config, overrides=_apply_eval_overrides(args))
        run_eval(cfg)
        return 0

    if args.command == "ablate-eval":
        overrides = list(args.overrides or [])
        _append_override(overrides, "experiment.name", args.run_name)
        _append_override(overrides, "experiment.seed", args.seed)
        _append_override(overrides, "eval.checkpoint", args.checkpoint)
        _append_override(overrides, "eval.action_stats_path", args.action_stats)
        _append_override(overrides, "eval.n_episodes", args.n_episodes)
        _append_override(overrides, "eval.max_steps", args.max_steps)
        _append_override(overrides, "eval.strict_parity", args.strict_parity)
        cfg = load_config(args.config, overrides=overrides)
        run_eval_ablation(
            cfg,
            execute_steps_values=_parse_csv_values(args.execute_steps_values, int),
            flow_steps_values=_parse_csv_values(args.n_flow_steps_values, int),
            smoothing_values=_parse_csv_values(args.smoothing_values, float),
        )
        return 0

    if args.command == "deploy-sim":
        from mini_pi0.deploy.sim_runner import run_deploy_sim

        cfg = load_config(args.config, overrides=_apply_deploy_sim_overrides(args))
        run_deploy_sim(cfg)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
