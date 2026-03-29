from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from mini_pi0.config.schema import ModelConfig, RootConfig, effective_image_keys
from mini_pi0.models.fm import MiniPi0FlowMatching

_MODEL_REGISTRY = {
    "mini_pi0_fm": MiniPi0FlowMatching,
}


def list_models() -> list[str]:
    """List model names registered in the repository model factory.

    Returns:
        Sorted model registry keys.
    """

    return sorted(_MODEL_REGISTRY.keys())


def make_model(model_cfg: ModelConfig | RootConfig) -> nn.Module:
    """Instantiate a model from typed model config.

    Args:
        model_cfg: ``ModelConfig`` or full ``RootConfig`` containing ``model`` section.

    Returns:
        Newly created torch module.

    Raises:
        ValueError: If ``model.name`` is unknown.
    """

    cfg = model_cfg.model if isinstance(model_cfg, RootConfig) else model_cfg
    key = str(cfg.name).strip().lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{cfg.name}'. Options: {list_models()}")
    cls = _MODEL_REGISTRY[key]
    return cls(
        action_dim=cfg.action_dim,
        prop_dim=cfg.prop_dim,
        obs_mode=cfg.obs_mode,
        vision_dim=cfg.vision_dim,
        chunk_size=cfg.chunk_size,
        cond_dim=cfg.cond_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        nlayers=cfg.nlayers,
    )


def count_params(module: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters for a module.

    Args:
        module: Torch module.

    Returns:
        ``(total_params, trainable_params)``.
    """

    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def pretty_print_model_tree(module: nn.Module, name: str = "model", indent: int = 0, max_depth: int = 3) -> None:
    """Print a readable module tree with parameter counts.

    Args:
        module: Root module to print.
        name: Display name of current module node.
        indent: Current indentation depth.
        max_depth: Max recursive child depth to print.
    """

    total, trainable = count_params(module)
    pad = "  " * indent
    print(f"{pad}- {name}: {module.__class__.__name__} | params={total:,} | trainable={trainable:,}")
    if indent >= max_depth:
        return
    for child_name, child in module.named_children():
        pretty_print_model_tree(child, name=child_name, indent=indent + 1, max_depth=max_depth)


def build_checkpoint_payload(
    *,
    model: nn.Module,
    cfg: RootConfig,
    epoch: int,
    loss: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical checkpoint payload for save/load compatibility.

    Args:
        model: Trained model.
        cfg: Resolved root config.
        epoch: Zero-based epoch index at save time.
        loss: Epoch loss value.
        extra: Optional extra metadata merged into payload.

    Returns:
        Dictionary suitable for ``torch.save``.
    """

    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "loss": float(loss),
        "model": model.state_dict(),
        "model_name": cfg.model.name,
        "model_config": {
            "action_dim": cfg.model.action_dim,
            "prop_dim": cfg.model.prop_dim,
            "obs_mode": cfg.model.obs_mode,
            "vision_dim": cfg.model.vision_dim,
            "chunk_size": cfg.model.chunk_size,
            "cond_dim": cfg.model.cond_dim,
            "d_model": cfg.model.d_model,
            "nhead": cfg.model.nhead,
            "nlayers": cfg.model.nlayers,
        },
        "sim_backend": cfg.simulator.backend,
        "sim_config": {
            "backend": cfg.simulator.backend,
            "task": cfg.simulator.task,
            "robot": cfg.simulator.robot,
            "controller": cfg.simulator.controller,
            "control_freq": cfg.simulator.control_freq,
            "horizon": cfg.simulator.horizon,
            "camera_names": list(cfg.simulator.camera_names),
        },
        "robot_config": {
            "name": cfg.robot.name,
            "action_dim": cfg.robot.action_dim,
            "image_key": cfg.robot.image_key,
            "image_keys": effective_image_keys(cfg.robot),
            "state_keys": list(cfg.robot.state_keys) if cfg.robot.state_keys is not None else None,
            "proprio_keys": list(cfg.robot.proprio_keys),
        },
        "data_config": {
            "format": cfg.data.format,
            "observation_mode": cfg.data.observation_mode,
            "robomimic_hdf5": cfg.data.robomimic_hdf5,
            "lerobot_repo_id": cfg.data.lerobot_repo_id,
            "precomputed_features_path": cfg.data.precomputed_features_path,
            "precomputed_feature_key": cfg.data.precomputed_feature_key,
        },
        "vision_config": {
            "backend": cfg.vision.backend,
            "model_name": cfg.vision.model_name,
            "pretrained": cfg.vision.pretrained,
            "batch_size": cfg.vision.batch_size,
            "image_size": cfg.vision.image_size,
            "output_path": cfg.vision.output_path,
            "use_runtime_extractor": cfg.vision.use_runtime_extractor,
            "hf_model_id": cfg.vision.hf_model_id,
            "local_files_only": cfg.vision.local_files_only,
        },
    }
    if extra:
        payload.update(extra)
    return payload


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Save checkpoint payload to disk.

    Args:
        path: Destination checkpoint path.
        payload: Checkpoint dictionary.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_checkpoint(path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
    """Load checkpoint payload from disk.

    Args:
        path: Checkpoint path.
        map_location: Optional torch map_location argument.

    Returns:
        Loaded checkpoint dictionary/state.
    """

    return torch.load(path, map_location=map_location)
