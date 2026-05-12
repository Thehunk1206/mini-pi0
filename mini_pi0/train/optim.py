from __future__ import annotations

"""Optimizer, scheduler, and EMA helpers for training."""

from typing import Any

import torch

from mini_pi0.config.schema import RootConfig
from mini_pi0.models.mini_pi05 import PI05SmolVLM


class ExponentialMovingAverage:
    """Maintain an exponential moving average copy of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: torch.nn.Module) -> None:
        """Update EMA state from current model parameters."""
        with torch.no_grad():
            current = model.state_dict()
            for k, v in current.items():
                if k not in self.shadow:
                    self.shadow[k] = v.detach().clone()
                    continue
                if torch.is_floating_point(v):
                    shadow = self.shadow[k]
                    if shadow.device != v.device or shadow.dtype != v.dtype:
                        shadow = shadow.to(device=v.device, dtype=v.dtype)
                        self.shadow[k] = shadow
                    shadow.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[k] = v.detach().clone()

    def copy_to(self, model: torch.nn.Module) -> None:
        """Copy EMA weights into model in-place."""
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict[str, Any]:
        """Return serializable EMA state."""
        return {
            "decay": float(self.decay),
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore EMA state from checkpoint payload."""
        if not isinstance(state, dict):
            return
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow")
        if isinstance(shadow, dict):
            self.shadow = {k: v.detach().clone() for k, v in shadow.items()}


def snapshot_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Create a detached clone of full model state dict."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def restore_model_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Restore model state from detached snapshot."""
    model.load_state_dict(state, strict=True)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: RootConfig,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, str]:
    """Create learning-rate scheduler from config."""
    kind = str(getattr(cfg.train, "lr_scheduler", "cosine")).strip().lower()
    if kind in {"none", "off", "disabled"}:
        return None, "None"

    if kind == "cosine":
        t_max = int(getattr(cfg.train, "scheduler_t_max", 0) or 0)
        if t_max <= 0:
            t_max = max(1, int(cfg.train.epochs))
        eta_min = float(getattr(cfg.train, "scheduler_eta_min", 0.0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
        return scheduler, f"CosineAnnealingLR(T_max={t_max}, eta_min={eta_min:g})"

    if kind == "step":
        step_size = max(1, int(getattr(cfg.train, "scheduler_step_size", 50)))
        gamma = float(getattr(cfg.train, "scheduler_gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        return scheduler, f"StepLR(step_size={step_size}, gamma={gamma:g})"

    raise ValueError(
        f"Unsupported train.lr_scheduler '{cfg.train.lr_scheduler}'. "
        "Supported: cosine, step, none."
    )


def build_optimizer(
    model: torch.nn.Module,
    cfg: RootConfig,
) -> tuple[torch.optim.Optimizer, dict[str, float]]:
    """Build optimizer with optional backbone/expert LR groups.

    ``lr_backbone`` targets the observation/VLM backbone. ``lr_expert`` targets
    the action denoiser/expert. For non-VLM policies, unset values fall back to
    the base LR to preserve the old single-LR behavior.
    """
    base_lr = float(cfg.train.lr)
    weight_decay = float(cfg.train.weight_decay)
    lr_backbone_cfg = getattr(cfg.train, "lr_backbone", None)
    lr_expert_cfg = getattr(cfg.train, "lr_expert", None)

    if isinstance(model, PI05SmolVLM):
        lr_backbone = float(lr_backbone_cfg) if lr_backbone_cfg is not None else base_lr * 0.1
        lr_expert = float(lr_expert_cfg) if lr_expert_cfg is not None else base_lr
        backbone_params: list[torch.nn.Parameter] = []
        expert_params: list[torch.nn.Parameter] = []
        for name, param in model.named_parameters():
            if name.startswith("backbone.smolvlm."):
                backbone_params.append(param)
            else:
                expert_params.append(param)

        param_groups: list[dict[str, Any]] = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr_backbone, "name": "backbone"})
        if expert_params:
            param_groups.append({"params": expert_params, "lr": lr_expert, "name": "expert"})
        if not param_groups:
            raise ValueError("No trainable parameter groups found for optimizer.")

        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        return optimizer, {"backbone_lr": lr_backbone, "expert_lr": lr_expert}

    obs_encoder = getattr(model, "obs_encoder", None)
    action_expert = getattr(model, "action_transformer", None)
    if isinstance(obs_encoder, torch.nn.Module) and isinstance(action_expert, torch.nn.Module):
        lr_backbone = float(lr_backbone_cfg) if lr_backbone_cfg is not None else base_lr
        lr_expert = float(lr_expert_cfg) if lr_expert_cfg is not None else base_lr

        backbone_params = [p for p in obs_encoder.parameters() if p.requires_grad]
        expert_params = [p for p in action_expert.parameters() if p.requires_grad]
        grouped_ids = {id(p) for p in backbone_params + expert_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in grouped_ids]

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr_backbone, "name": "backbone"})
        if expert_params:
            param_groups.append({"params": expert_params, "lr": lr_expert, "name": "expert"})
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr, "name": "other"})
        if not param_groups:
            raise ValueError("No trainable parameter groups found for optimizer.")

        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        return optimizer, {"backbone_lr": lr_backbone, "expert_lr": lr_expert}

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    return optimizer, {"backbone_lr": base_lr, "expert_lr": base_lr}
