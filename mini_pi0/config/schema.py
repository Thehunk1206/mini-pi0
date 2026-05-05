from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    """Experiment-level metadata and artifact layout options.

    Attributes:
        name: Human-readable run group name used in output directory naming.
        seed: Random seed propagated to training and evaluation entrypoints.
        runs_root: Root directory under which timestamped run folders are created.
    """

    name: str = "mini-pi0"
    seed: int = 0
    runs_root: str = "runs"


@dataclass
class SimulatorConfig:
    """Simulator backend and environment instantiation settings.

    Attributes:
        backend: Simulator backend key (for example ``robosuite``).
        task: Task name understood by the selected backend.
        robot: Robot name/model for the configured task.
        controller: Controller profile name for backends that support controller presets.
        control_freq: Control rate in Hz for environment stepping.
        horizon: Maximum episode length in environment steps.
        reward_shaping: Enables shaped rewards when backend supports it.
        has_renderer: Enables interactive renderer window if available.
        has_offscreen_renderer: Enables offscreen rendering for recording and metrics.
        use_camera_obs: Enables image observations in environment observations.
        camera_names: Ordered camera names; first entry is used as default render camera.
        camera_width: Width for camera observations.
        camera_height: Height for camera observations.
        env_kwargs: Arbitrary backend-specific kwargs passed at environment creation time.
    """

    backend: str = "robosuite"  # robosuite | maniskill3 | isaaclab
    task: str = "Lift"
    robot: str = "Panda"
    controller: str = "BASIC"
    control_freq: int = 20
    horizon: int = 400
    reward_shaping: bool = True
    has_renderer: bool = False
    has_offscreen_renderer: bool = True
    use_camera_obs: bool = True
    camera_names: list[str] = field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])
    camera_width: int = 84
    camera_height: int = 84
    env_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Task-specific object controls used during evaluation/deployment.

    Attributes:
        object_name: Semantic object name used by adapter hooks such as pose overrides.
        object_pose_override: Optional backend-specific pose dictionary.
    """

    object_name: str = "cube"
    object_pose_override: dict[str, Any] | None = None


@dataclass
class RobotConfig:
    """Canonical robot observation/action schema used by model code.

    Attributes:
        name: Robot identifier for dataset metadata and adapter wiring.
        action_dim: Expected action vector dimensionality.
        proprio_keys: Observation keys concatenated into proprioceptive state vectors.
        state_keys: Optional alias for the policy state vector keys. When provided,
            this list is used in place of ``proprio_keys`` across train/eval/deploy.
        image_key: Backward-compatible single observation image key used as model image input.
        image_keys: Optional ordered list of image observation keys. When set, this list
            is used everywhere instead of ``image_key`` and enables multi-camera conditioning.
    """

    name: str = "Panda"
    action_dim: int = 7
    proprio_keys: list[str] = field(
        default_factory=lambda: ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    )
    state_keys: list[str] | None = None
    image_key: str = "agentview_image"
    image_keys: list[str] | None = None


def effective_state_keys(robot: RobotConfig) -> list[str]:
    """Resolve state feature keys used by policy models.

    Args:
        robot: Robot section from root config.

    Returns:
        Ordered list of observation keys used as state input.
    """

    if robot.state_keys is not None and len(robot.state_keys) > 0:
        return list(robot.state_keys)
    return list(robot.proprio_keys)


def effective_image_keys(robot: RobotConfig) -> list[str]:
    """Resolve image observation keys used by data/sim pipelines.

    Args:
        robot: Robot section from root config.

    Returns:
        Ordered list of image observation keys. Falls back to ``image_key`` when
        ``image_keys`` is empty or unset.
    """

    if robot.image_keys is not None:
        keys = [str(k).strip() for k in robot.image_keys if str(k).strip()]
        if keys:
            return keys
    return [str(robot.image_key)]


@dataclass
class DataConfig:
    """Dataset loading and preprocessing configuration.

    Attributes:
        format: Dataset format identifier (``robomimic_hdf5`` or ``lerobot_hf``).
        observation_mode: Observation payload mode consumed by policy model.
            ``image`` uses raw image tensors; ``precomputed`` uses cached feature vectors.
        robomimic_hdf5: Path to robomimic HDF5 file for ``robomimic_hdf5`` format.
        robomimic_data_group: Top-level HDF5 group containing demonstrations.
        lerobot_repo_id: Hugging Face LeRobot dataset repo id for ``lerobot_hf``.
        lerobot_action_key: Sample key used as action vector in LeRobot samples.
        lerobot_episode_index_key: Sample key used for episode grouping in LeRobot samples.
        lerobot_local_files_only: Load LeRobot dataset from local cache only.
        lerobot_video_backend: Video decoding backend for LeRobot frames (`pyav`, `torchcodec`, or `video_reader`).
        precomputed_features_path: Path to `.npz` with precomputed per-episode vision features.
        precomputed_feature_key: Observation key used to store/read cached features.
        fallback_image_hw: Fallback ``[H, W]`` used when source frames are missing/invalid.
        n_demos: Optional max number of demos to load.
        chunk_size: Action horizon length per supervised training sample.
        action_stats_path: Output/input path for action normalization statistics.
        filter_min_episode_length: Drop episodes shorter than this many timesteps (0 disables).
        filter_min_action_std: Drop episodes with action std below this threshold (0 disables).
        filter_min_state_delta: Drop episodes with low start/end state movement (0 disables).
        filter_state_delta_key: Preferred observation key used for state-progress filtering.
        filter_drop_nan: Drop episodes containing NaN/Inf values in observations or actions.
    """

    format: str = "robomimic_hdf5"
    observation_mode: str = "image"  # image | precomputed
    robomimic_hdf5: str | None = "data/robomimic/lift/ph/low_dim_v15.hdf5"
    robomimic_data_group: str = "data"
    lerobot_repo_id: str | None = None
    lerobot_action_key: str = "action"
    lerobot_episode_index_key: str = "episode_index"
    lerobot_local_files_only: bool = False
    lerobot_video_backend: str | None = "pyav"
    precomputed_features_path: str | None = None
    precomputed_feature_key: str = "vision_feat"
    fallback_image_hw: list[int] = field(default_factory=lambda: [84, 84])
    n_demos: int | None = 200
    chunk_size: int = 16
    action_stats_path: str = "action_stats.json"
    filter_min_episode_length: int = 0
    filter_min_action_std: float = 0.0
    filter_min_state_delta: float = 0.0
    filter_state_delta_key: str | None = "observation.state.object"
    filter_drop_nan: bool = True


@dataclass
class DatasetCollectionConfig:
    """High-level oracle dataset mixture collection settings.

    Attributes:
        output_hdf5: Destination robomimic-style HDF5 path.
        total_episodes: Number of accepted success-only episodes to write.
        num_envs: Number of parallel ManiSkill envs for scripted collection.
        max_steps: Per-trial rollout budget.
        only_success: Whether to retain only successful rollouts.
        reject_long_episodes: Reject trajectories above a profile length limit.
        mix: Ratios for core, recovery, and suboptimal demos.
        difficulty: Internal perturbation/noise preset.
    """

    output_hdf5: str = "artifacts/physical_scripted_oracle_mix_500_obj2to4_h700_640.hdf5"
    total_episodes: int = 500
    num_envs: int = 1
    max_steps: int = 700
    only_success: bool = True
    reject_long_episodes: bool = True
    mix: dict[str, float] = field(default_factory=lambda: {"core": 0.65, "recovery": 0.25, "suboptimal": 0.10})
    difficulty: str = "balanced"
    force_perturbation_type: str | None = None


@dataclass
class ModelConfig:
    """Flow-matching action model architecture configuration.

    Attributes:
        name: Model registry key.
        action_dim: Output action dimensionality.
        prop_dim: Input proprioceptive vector dimensionality.
        obs_mode: Observation input mode for model encoder (``image`` or ``feature``).
        vision_dim: Input feature dimension when ``obs_mode=feature``.
        chunk_size: Number of actions generated per model sample call.
        cond_dim: Conditioning feature size from observation encoder.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads per transformer layer.
        nlayers: Number of transformer encoder layers.
        num_timestep_buckets: Number of discrete diffusion timestep buckets.
        noise_beta_alpha: Alpha parameter for Beta timestep sampling distribution.
        noise_beta_beta: Beta parameter for Beta timestep sampling distribution.
        noise_s: Upper timestep cutoff multiplier used in sampling.
        state_dropout_prob: Probability of dropping state token during training.
        state_additive_noise_scale: Gaussian noise scale added to state token during training.
        add_action_pos_embed: Enable learned positional embeddings for action tokens.
        use_context_layernorm: Apply LayerNorm to context tokens before denoiser.
        vision_token_grid_size: Spatial token grid size per side for image-mode context.
        use_dit_adaln: Enable DiT-style timestep-conditioned AdaLN modulation in denoiser blocks.
        pretrained_model_name_or_path: Optional SmolVLM checkpoint id/path used when
            ``name=mini_pi05``. If null, ``mini_pi05`` starts from random init.
        pretrained_variant: Backbone shape hint used for ``mini_pi05`` pretrained
            loading (``256M`` or ``500M``).
        pretrained_local_files_only: Restrict Hugging Face loading to local cache only.
        expert_intermediate_size: Optional MiniPI05 action-expert MLP width.
            Attention structure remains locked to the SmolVLM text backbone, but
            the expert MLP can be narrowed to reduce parameters and inference cost.
        dtype: Runtime dtype for models that support it, especially ``mini_pi05``
            (``float32`` or ``bfloat16``).
    """

    name: str = "mini_pi0_fm"
    action_dim: int = 7
    prop_dim: int = 9
    obs_mode: str = "image"  # image | feature
    vision_dim: int = 0
    chunk_size: int = 16
    cond_dim: int = 256
    d_model: int = 256
    nhead: int = 4
    nlayers: int = 4
    num_timestep_buckets: int = 1000
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    state_dropout_prob: float = 0.0
    state_additive_noise_scale: float = 0.0
    add_action_pos_embed: bool = True
    use_context_layernorm: bool = True
    vision_token_grid_size: int = 4
    use_dit_adaln: bool = True
    pretrained_model_name_or_path: str | None = None
    pretrained_variant: str = "256M"
    pretrained_local_files_only: bool = False
    expert_intermediate_size: int | None = None
    dtype: str | None = None


@dataclass
class VisionConfig:
    """Vision encoder configuration for precompute and runtime feature extraction.

    Attributes:
        backend: Encoder backend (`torchvision`, `timm`, or `hf`).
        model_name: Backbone identifier for selected backend.
        pretrained: Load pretrained weights when backend supports it.
        batch_size: Batch size used during offline feature extraction.
        image_size: Input resize/crop size for encoder.
        output_path: Default output path for precomputed feature `.npz`.
        use_runtime_extractor: Compute features online during eval/deploy when model expects features.
        hf_model_id: Optional HF model id (used when backend=`hf`).
        local_files_only: Restrict HF model loading to local cache.
    """

    backend: str = "timm"
    model_name: str = "vit_small_patch14_dinov2.lvd142m"
    pretrained: bool = True
    batch_size: int = 64
    image_size: int = 224
    output_path: str = "data/features/vision_features.npz"
    use_runtime_extractor: bool = True
    hf_model_id: str | None = None
    local_files_only: bool = False


@dataclass
class TrainConfig:
    """Training loop hyperparameters and runtime controls.

    Attributes:
        epochs: Number of full dataset passes.
        batch_size: Minibatch size for optimization.
        lr: Optimizer learning rate.
        lr_scheduler: Learning-rate scheduler type (`cosine`, `step`, or `none`).
        scheduler_t_max: Optional cosine scheduler period. Defaults to ``epochs`` when null.
        scheduler_eta_min: Minimum learning rate for cosine scheduler.
        scheduler_step_size: Step period (epochs) for step scheduler.
        scheduler_gamma: Multiplicative LR decay factor for step scheduler.
        weight_decay: AdamW weight decay.
        num_workers: DataLoader worker count (-1 means auto).
        persistent_workers: Keep DataLoader workers alive between epochs.
        save_best: Save checkpoints when loss improves.
        save_best_min_delta: Minimum loss improvement required to trigger save.
        grad_clip_norm: Gradient clipping norm threshold (<=0 disables clipping).
        model_print_depth: Depth for model architecture pretty-print tree.
        resume_from: Optional checkpoint path to resume training from.
        resume_optimizer: Restore optimizer/scheduler states when available.
        val_ratio: Fraction of training samples reserved for validation (0 disables validation).
        ema_decay: Exponential moving average decay for model weights (0 disables EMA).
        checkpoint_use_ema: Save EMA weights into checkpoint model payload when EMA is enabled.
        lr_backbone: Optional override LR for pretrained VLM backbone params.
            When null, falls back to ``lr * 0.1`` for ``mini_pi05``.
        lr_expert: Optional override LR for action expert/head params.
            When null, falls back to ``lr`` for ``mini_pi05``.
        freeze_backbone_steps: Number of initial optimizer steps to keep
            ``mini_pi05`` VLM backbone frozen before unfreezing for finetuning.
        image_aug_enable: Enable training-time image augmentation.
        image_aug_crop_scale: Random resized-crop scale lower bound in (0, 1].
            1.0 disables random crop.
        image_aug_brightness: Symmetric brightness jitter magnitude.
            Final factor sampled in ``[1-x, 1+x]``.
        image_aug_contrast: Symmetric contrast jitter magnitude.
        image_aug_saturation: Symmetric saturation jitter magnitude.
        action_noise_std: Standard deviation of Gaussian action noise injected
            into training targets (normalized action space).
        action_noise_clip: Optional absolute clipping after action noise.
            0 disables clipping.
        sim_eval_every_epochs: Run simulation rollout eval every N epochs.
            ``0`` disables training-time sim eval.
        sim_eval_n_episodes: Number of rollout episodes for training-time sim eval.
        sim_eval_max_steps: Rollout step budget for training-time sim eval.
        sim_eval_record_grid: Save success/failure grid videos during sim eval.
        save_best_success: Save ``best_success.pt`` when sim success improves.
        save_best_success_min_rate: Minimum success rate required before writing
            ``best_success.pt``. Keeps a 0% rollout from being labeled "best".
        device: Requested torch device (``auto``, ``cpu``, ``cuda``, ``mps``).
    """

    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-4
    lr_scheduler: str = "cosine"
    scheduler_t_max: int | None = None
    scheduler_eta_min: float = 0.0
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.5
    weight_decay: float = 1e-4
    num_workers: int = -1
    persistent_workers: bool = True
    save_best: bool = True
    save_best_min_delta: float = 1e-4
    grad_clip_norm: float = 1.0
    model_print_depth: int = 3
    resume_from: str | None = None
    resume_optimizer: bool = True
    val_ratio: float = 0.1
    ema_decay: float = 0.0
    checkpoint_use_ema: bool = True
    lr_backbone: float | None = None
    lr_expert: float | None = None
    freeze_backbone_steps: int = 0
    image_aug_enable: bool = False
    image_aug_crop_scale: float = 1.0
    image_aug_brightness: float = 0.0
    image_aug_contrast: float = 0.0
    image_aug_saturation: float = 0.0
    action_noise_std: float = 0.0
    action_noise_clip: float = 0.0
    sim_eval_every_epochs: int = 0
    sim_eval_n_episodes: int = 10
    sim_eval_max_steps: int | None = None
    sim_eval_record_grid: bool = False
    save_best_success: bool = True
    save_best_success_min_rate: float = 0.0
    device: str = "auto"


@dataclass
class EvalConfig:
    """Evaluation loop configuration and reporting controls.

    Attributes:
        run_dir: Optional run directory to write eval artifacts into. If ``None``,
            eval will reuse checkpoint run dir when discoverable, otherwise create a new run.
        checkpoint: Path to model checkpoint used for evaluation.
        action_stats_path: Path to action normalization stats.
        n_episodes: Number of rollout episodes.
        execute_steps: Actions executed from each predicted chunk before re-planning.
        n_flow_steps: Euler integration steps used in flow-matching sampling.
        max_steps: Optional hard step limit per episode.
        verbose: Enables live progress logging during eval.
        log_every_episodes: Episode log interval when ``verbose=True``.
        record: Enables per-episode rollout recording.
        record_grid: Enables success/failure grid video export.
        grid_size: Grid dimension (``N`` creates ``N x N`` tiles).
        grid_fps: Frames per second for grid videos.
        grid_width: Per-tile frame width in pixels.
        grid_height: Per-tile frame height in pixels.
        cube_xy: Optional fixed cube x/y position override.
        cube_xy_range: Optional randomized cube x/y range ``[xmin, xmax, ymin, ymax]``.
        cube_z: Optional fixed cube z override.
        cube_yaw_deg: Optional fixed cube yaw override in degrees.
        plot_path: Legacy output path for metrics plot copy.
        strict_parity: Enforce checkpoint/runtime parity checks before rollout.
        action_smoothing_alpha: Action exponential smoothing factor in ``[0, 1]`` (0 disables).
        action_scale: Optional per-dimension multiplicative scale applied before clipping.
        disable_domain_randomization: Disable simulator domain randomization during
            policy eval. This gives a stable held-out task metric; robustness eval
            can explicitly set it to ``False``.
        vectorized: Enable ManiSkill vectorized rollout evaluation when supported.
        num_envs: Number of parallel ManiSkill environments for vectorized eval.
        failure_reward_threshold: Threshold for classifying failures as ``no_progress``.
        stability_warmup_steps: Number of initial env steps using warmup rollout controls.
        stability_warmup_execute_steps: Warmup override for execute steps (null keeps base value).
        stability_warmup_n_flow_steps: Warmup override for denoising flow steps.
        stability_warmup_action_smoothing_alpha: Warmup override for action smoothing alpha.
        device: Requested torch device for evaluation.
    """

    run_dir: str | None = None
    checkpoint: str = "checkpoints/best.pt"
    action_stats_path: str = "action_stats.json"
    n_episodes: int = 50
    execute_steps: int = 8
    n_flow_steps: int = 10
    max_steps: int | None = None
    verbose: bool = True
    log_every_episodes: int = 1
    record: bool = False
    record_grid: bool = False
    grid_size: int = 3
    grid_fps: int = 20
    grid_width: int = 256
    grid_height: int = 256
    cube_xy: list[float] | None = None
    cube_xy_range: list[float] | None = None
    cube_z: float | None = None
    cube_yaw_deg: float | None = None
    plot_path: str = "eval_metrics.png"
    strict_parity: bool = True
    action_smoothing_alpha: float = 0.0
    action_scale: list[float] | None = None
    disable_domain_randomization: bool = True
    vectorized: bool = False
    num_envs: int = 1
    failure_reward_threshold: float = 0.2
    stability_warmup_steps: int = 0
    stability_warmup_execute_steps: int | None = None
    stability_warmup_n_flow_steps: int | None = None
    stability_warmup_action_smoothing_alpha: float | None = None
    device: str = "auto"


@dataclass
class DeployConfig:
    """Deployment configuration for simulation/hardware execution loops.

    Attributes:
        mode: Deployment mode (currently ``sim`` in modular package).
        checkpoint: Path to model checkpoint.
        action_stats_path: Path to action normalization stats.
        execute_steps: Number of predicted actions executed before re-planning.
        n_flow_steps: Flow sampler integration steps.
        max_steps: Maximum loop iterations per deploy run.
        record_path: Optional output video path.
        strict_parity: Enforce checkpoint/runtime parity checks before rollout.
        action_smoothing_alpha: Action exponential smoothing factor in ``[0, 1]`` (0 disables).
        action_scale: Optional per-dimension multiplicative scale applied before clipping.
        stability_warmup_steps: Number of initial env steps using warmup rollout controls.
        stability_warmup_execute_steps: Warmup override for execute steps (null keeps base value).
        stability_warmup_n_flow_steps: Warmup override for denoising flow steps.
        stability_warmup_action_smoothing_alpha: Warmup override for action smoothing alpha.
        device: Requested torch device.
    """

    mode: str = "sim"
    checkpoint: str = "checkpoints/best.pt"
    action_stats_path: str = "action_stats.json"
    execute_steps: int = 4
    n_flow_steps: int = 10
    max_steps: int = 500
    record_path: str | None = None
    strict_parity: bool = True
    action_smoothing_alpha: float = 0.0
    action_scale: list[float] | None = None
    stability_warmup_steps: int = 0
    stability_warmup_execute_steps: int | None = None
    stability_warmup_n_flow_steps: int | None = None
    stability_warmup_action_smoothing_alpha: float | None = None
    device: str = "auto"


@dataclass
class RootConfig:
    """Top-level strongly typed configuration object for all CLI commands.

    Attributes:
        experiment: Global experiment/run metadata.
        simulator: Simulator backend/task options.
        task: Task-level object randomization and overrides.
        robot: Canonical robot schema.
        data: Dataset and preprocessing settings.
        dataset_collection: Oracle mixture collection settings.
        vision: Vision encoder settings for feature precompute/runtime extraction.
        model: Model architecture settings.
        train: Training loop controls.
        eval: Evaluation loop controls.
        deploy: Deployment loop controls.
    """

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dataset_collection: DatasetCollectionConfig = field(default_factory=DatasetCollectionConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)


def to_dict(cfg: RootConfig) -> dict[str, Any]:
    """Convert the typed root config dataclass tree into a plain dictionary.

    Args:
        cfg: Root configuration dataclass object.

    Returns:
        JSON/YAML-serializable dictionary representation.
    """

    return asdict(cfg)
