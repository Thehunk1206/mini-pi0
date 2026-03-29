from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mini_pi0.config.io import dump_config
from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys, to_dict
from mini_pi0.dataset.episodes import load_episodes_from_config
from mini_pi0.dataset.stats import ActionStats
from mini_pi0.dataset.torch_dataset import ActionChunkDataset
from mini_pi0.models.registry import (
    build_checkpoint_payload,
    count_params,
    load_checkpoint,
    make_model,
    pretty_print_model_tree,
    save_checkpoint,
)
from mini_pi0.utils.device import resolve_device
from mini_pi0.utils.runs import append_jsonl, create_run_dir

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducibility.

    Args:
        seed: Global random seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_num_workers(value: int) -> int:
    """Resolve configured DataLoader workers, supporting ``-1`` auto mode.

    Args:
        value: Requested worker count (negative means auto).

    Returns:
        Safe worker count for this machine.
    """

    if value is None or int(value) < 0:
        return int(min(4, os.cpu_count() or 1))
    return int(max(0, int(value)))


def _infer_action_dim(episodes) -> int:
    """Infer a consistent action dimension from loaded episodes."""

    dims = {int(np.asarray(ep.actions).shape[-1]) for ep in episodes if np.asarray(ep.actions).ndim == 2}
    if not dims:
        raise ValueError("Unable to infer action dimension from dataset episodes.")
    if len(dims) != 1:
        raise ValueError(f"Inconsistent action dimensions across episodes: {sorted(dims)}")
    return int(next(iter(dims)))


def _infer_prop_dim(obs: dict[str, np.ndarray], proprio_keys: list[str]) -> int:
    """Infer concatenated proprio dimension from one observation sample."""

    return int(sum(np.asarray(obs[k], dtype=np.float32).reshape(-1).shape[0] for k in proprio_keys))


def _infer_visual_mode_and_dim(
    obs: dict[str, np.ndarray],
    observation_key: str | None,
    image_keys: list[str],
) -> tuple[str, int]:
    """Infer visual input mode and dimension from one observation sample."""

    if observation_key is not None:
        visual = np.asarray(obs[observation_key])
    elif len(image_keys) == 1:
        visual = np.asarray(obs[image_keys[0]])
    else:
        visual_parts = [np.asarray(obs[k]) for k in image_keys]
        if all(v.ndim == 1 for v in visual_parts):
            return "feature", int(sum(int(v.reshape(-1).shape[0]) for v in visual_parts))
        if all(v.ndim >= 2 for v in visual_parts):
            h, w = visual_parts[0].shape[:2]
            c = visual_parts[0].shape[2] if visual_parts[0].ndim >= 3 else 1
            for idx, part in enumerate(visual_parts[1:], start=1):
                part_c = part.shape[2] if part.ndim >= 3 else 1
                if part.shape[:2] != (h, w) or part_c != c:
                    raise ValueError(
                        f"All image_keys must share spatial shape and channels for image fusion. "
                        f"Got {visual_parts[0].shape} and {part.shape} at index {idx}."
                    )
            return "image", 0
        raise ValueError(
            "Mixed visual tensor ranks across image_keys are not supported. "
            f"Shapes: {[tuple(v.shape) for v in visual_parts]}"
        )
    if visual.ndim == 1:
        return "feature", int(visual.shape[0])
    if visual.ndim == 3:
        return "image", 0
    raise ValueError(
        f"Unsupported visual observation shape {visual.shape} for key '{observation_key}'. "
        "Expected feature [D] or image [H,W,C]."
    )


class ExponentialMovingAverage:
    """Maintain an exponential moving average copy of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float):
        """Initialize EMA state.

        Args:
            model: Source model to track.
            decay: EMA decay factor in [0, 1).
        """

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
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
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
        decay = state.get("decay", self.decay)
        self.decay = float(decay)
        shadow = state.get("shadow")
        if isinstance(shadow, dict):
            self.shadow = {k: v.detach().clone() for k, v in shadow.items()}


def _snapshot_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Create a detached clone of full model state dict."""

    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _restore_model_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Restore model state from detached snapshot."""

    model.load_state_dict(state, strict=True)


def _split_train_val(dataset, val_ratio: float, seed: int) -> tuple[Any, Any | None]:
    """Split dataset into train and validation subsets.

    Args:
        dataset: Source torch dataset.
        val_ratio: Fraction to reserve for validation.
        seed: Split RNG seed.

    Returns:
        Tuple of (train_dataset, val_dataset_or_none).
    """

    ratio = float(max(0.0, min(0.9, val_ratio)))
    n_total = int(len(dataset))
    if ratio <= 0.0 or n_total < 2:
        return dataset, None

    n_val = int(round(n_total * ratio))
    n_val = max(1, min(n_total - 1, n_val))
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(n_total).tolist()
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: RootConfig,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, str]:
    """Create learning-rate scheduler from config.

    Args:
        optimizer: Optimizer instance to schedule.
        cfg: Resolved root configuration.

    Returns:
        Tuple of ``(scheduler_or_none, human_readable_description)``.
    """

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


def _print_header(cfg: RootConfig, resolved_device: torch.device, n_episodes: int, n_samples: int, model: torch.nn.Module) -> None:
    """Print human-readable training configuration and model summary.

    Args:
        cfg: Resolved root configuration.
        resolved_device: Device selected for training.
        n_episodes: Number of loaded demonstration episodes.
        n_samples: Number of chunked training samples.
        model: Instantiated model module.
    """

    total_params, trainable_params = count_params(model)

    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    cfg_dict = to_dict(cfg)
    cfg_dict["train"]["resolved_device"] = str(resolved_device)
    print(json.dumps(cfg_dict, indent=2, sort_keys=True))
    print("-" * 80)
    print(f"Episodes loaded      : {n_episodes}")
    print(f"Training samples     : {n_samples}")
    print(f"Model params (total) : {total_params:,}")
    print(f"Model params (train) : {trainable_params:,}")
    print("-" * 80)
    print("Model Architecture (Pretty Tree)")
    print("-" * 80)
    pretty_print_model_tree(model, max_depth=max(0, int(cfg.train.model_print_depth)))
    print("=" * 80)


def run_train(cfg: RootConfig) -> dict[str, Any]:
    """Execute end-to-end supervised training for the configured action model.

    Args:
        cfg: Resolved root configuration.

    Returns:
        Summary dictionary with run paths and best checkpoint metadata.
    """

    run_dir = create_run_dir(cfg.experiment.runs_root, cfg.experiment.name)
    dump_config(run_dir / "config_resolved.yaml", cfg)

    _seed_everything(int(cfg.experiment.seed))

    print(
        "[train] Loading dataset | "
        f"format={cfg.data.format} observation_mode={cfg.data.observation_mode} n_demos={cfg.data.n_demos}",
        flush=True,
    )
    episodes = load_episodes_from_config(cfg)
    print(f"[train] Dataset loaded | episodes={len(episodes)}", flush=True)
    state_keys = effective_state_keys(cfg.robot)
    image_keys = effective_image_keys(cfg.robot)
    obs_mode_cfg = str(getattr(cfg.data, "observation_mode", "image")).strip().lower()
    observation_key = (
        cfg.data.precomputed_feature_key if obs_mode_cfg in {"precomputed", "feature", "features"} else None
    )

    inferred_action_dim = _infer_action_dim(episodes)
    inferred_prop_dim = _infer_prop_dim(episodes[0].obs[0], state_keys)
    inferred_obs_mode, inferred_vision_dim = _infer_visual_mode_and_dim(
        episodes[0].obs[0],
        observation_key=observation_key,
        image_keys=image_keys,
    )

    if cfg.robot.action_dim != inferred_action_dim:
        print(
            f"[train] Overriding robot.action_dim from {cfg.robot.action_dim} to inferred {inferred_action_dim} "
            f"based on dataset actions."
        )
    if cfg.model.action_dim != inferred_action_dim:
        print(
            f"[train] Overriding model.action_dim from {cfg.model.action_dim} to inferred {inferred_action_dim} "
            f"based on dataset actions."
        )
    if cfg.model.prop_dim != inferred_prop_dim:
        print(
            f"[train] Overriding model.prop_dim from {cfg.model.prop_dim} to inferred {inferred_prop_dim} "
            f"based on dataset state keys."
        )
    if cfg.model.obs_mode != inferred_obs_mode:
        print(
            f"[train] Overriding model.obs_mode from {cfg.model.obs_mode} to inferred {inferred_obs_mode} "
            f"based on observation keys {([observation_key] if observation_key else image_keys)}."
        )
    if inferred_obs_mode == "feature" and cfg.model.vision_dim != inferred_vision_dim:
        print(
            f"[train] Overriding model.vision_dim from {cfg.model.vision_dim} to inferred {inferred_vision_dim} "
            f"based on cached vision features."
        )

    cfg.robot.action_dim = inferred_action_dim
    cfg.model.action_dim = inferred_action_dim
    cfg.model.prop_dim = inferred_prop_dim
    cfg.model.obs_mode = inferred_obs_mode
    cfg.model.vision_dim = inferred_vision_dim if inferred_obs_mode == "feature" else 0
    dump_config(run_dir / "config_resolved.yaml", cfg)

    all_actions = np.concatenate([ep.actions.astype(np.float32) for ep in episodes], axis=0)
    stats = ActionStats.from_actions(all_actions)

    # Canonical stats artifact for this run.
    run_stats_path = run_dir / "artifacts" / "action_stats.json"
    stats.save(str(run_stats_path))

    # Backward-compatible stats path used by wrappers and older scripts.
    legacy_stats_path = Path(cfg.data.action_stats_path)
    stats.save(str(legacy_stats_path))

    dataset = ActionChunkDataset(
        episodes=episodes,
        chunk_size=cfg.data.chunk_size,
        image_key=cfg.robot.image_key,
        image_keys=image_keys,
        proprio_keys=state_keys,
        action_stats=stats,
        observation_key=observation_key,
    )
    train_dataset, val_dataset = _split_train_val(
        dataset,
        val_ratio=float(getattr(cfg.train, "val_ratio", 0.0)),
        seed=int(cfg.experiment.seed),
    )
    print(
        f"[train] Prepared action-chunk dataset | total={len(dataset)} train={len(train_dataset)} "
        f"val={(len(val_dataset) if val_dataset is not None else 0)}",
        flush=True,
    )

    model = make_model(cfg)
    device = resolve_device(cfg.train.device)
    model = model.to(device)

    _print_header(cfg, device, n_episodes=len(episodes), n_samples=len(train_dataset), model=model)

    num_workers = _resolve_num_workers(cfg.train.num_workers)
    use_persistent = bool(cfg.train.persistent_workers and num_workers > 0)
    loader_kwargs = dict(
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=use_persistent,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = None
    if val_dataset is not None:
        val_kwargs = dict(loader_kwargs)
        val_kwargs["shuffle"] = False
        val_loader = DataLoader(val_dataset, **val_kwargs)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler, scheduler_desc = _build_scheduler(optimizer, cfg)
    ema_decay = float(getattr(cfg.train, "ema_decay", 0.0))
    ema = ExponentialMovingAverage(model, decay=ema_decay) if ema_decay > 0.0 else None

    start_epoch = 0
    best_metric = float("inf")
    best_metric_name = "val_loss" if val_loader is not None else "train_loss"
    resume_from = getattr(cfg.train, "resume_from", None)
    if resume_from:
        ckpt_path = Path(str(resume_from))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        print(f"[train] Resuming from checkpoint: {ckpt_path}", flush=True)
        ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)

        prev_epoch = int(ckpt.get("epoch", -1))
        start_epoch = max(0, prev_epoch + 1)
        best_metric = float(
            ckpt.get(
                "best_metric",
                ckpt.get("best_loss", ckpt.get("val_loss", ckpt.get("loss", float("inf")))),
            )
        )

        if bool(getattr(cfg.train, "resume_optimizer", True)):
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            else:
                print("[train] Resume note: optimizer state not found in checkpoint; using fresh optimizer.", flush=True)
            if scheduler is not None:
                if "scheduler" in ckpt and ckpt["scheduler"] is not None:
                    scheduler.load_state_dict(ckpt["scheduler"])
                else:
                    print("[train] Resume note: scheduler state not found in checkpoint; using fresh scheduler.", flush=True)
        if ema is not None and "ema" in ckpt and ckpt["ema"] is not None:
            try:
                ema.load_state_dict(ckpt["ema"])
            except Exception:
                print("[train] Resume note: failed to load EMA state; continuing with fresh EMA.", flush=True)

        print(
            f"[train] Resume state | start_epoch={start_epoch + 1} "
            f"previous_epoch={prev_epoch + 1} best_{best_metric_name}={best_metric:.6f}",
            flush=True,
        )

    print(f"Training on device : {device}")
    print(
        f"Dataloader         : batch_size={cfg.train.batch_size}, num_workers={num_workers}, "
        f"pin_memory={device.type == 'cuda'}, persistent_workers={use_persistent}"
    )
    print(
        "Validation         : "
        f"enabled={val_loader is not None}, val_ratio={float(getattr(cfg.train, 'val_ratio', 0.0)):.3f}, "
        f"val_batches={(len(val_loader) if val_loader is not None else 0)}"
    )
    print(f"Optimizer          : AdamW(lr={cfg.train.lr}, weight_decay={cfg.train.weight_decay})")
    print(f"Scheduler          : {scheduler_desc}")
    print(
        "EMA                : "
        f"enabled={ema is not None}, decay={float(getattr(cfg.train, 'ema_decay', 0.0)):.6f}, "
        f"checkpoint_use_ema={bool(getattr(cfg.train, 'checkpoint_use_ema', True))}"
    )
    print(f"Checkpointing      : save_best={cfg.train.save_best}, min_delta={cfg.train.save_best_min_delta}")
    print(f"Resume             : checkpoint={resume_from}, restore_opt={bool(getattr(cfg.train, 'resume_optimizer', True))}")
    print(f"Batches / epoch    : {len(loader)}")

    run_ckpt_dir = run_dir / "checkpoints"
    if resume_from:
        try:
            ckpt = load_checkpoint(str(resume_from), map_location="cpu")
            save_checkpoint(run_ckpt_dir / "best.pt", ckpt)
        except Exception:
            pass

    epochs = int(cfg.train.epochs)
    if start_epoch >= epochs:
        print(
            f"[train] Requested epochs={epochs}, resume starts at epoch={start_epoch + 1}. Nothing to run.",
            flush=True,
        )
        if resume_from:
            try:
                ckpt = load_checkpoint(str(resume_from), map_location="cpu")
                save_checkpoint(run_ckpt_dir / "best.pt", ckpt)
            except Exception:
                pass
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        epoch_start = time.perf_counter()
        data_wait_s = 0.0
        compute_s = 0.0
        save_s = 0.0

        iterator = loader
        t_setup0 = time.perf_counter()
        if tqdm is not None:
            iterator = tqdm(
                loader,
                desc=f"Epoch {epoch + 1:03d}/{epochs:03d}",
                leave=False,
                dynamic_ncols=True,
            )
        iter_setup_s = time.perf_counter() - t_setup0

        iter_end = time.perf_counter()
        for step, (img, prop, actions) in enumerate(iterator, start=1):
            batch_ready = time.perf_counter()
            data_wait_s += batch_ready - iter_end

            step_start = time.perf_counter()
            img = img.to(device)
            prop = prop.to(device)
            actions = actions.to(device)

            loss = model(img, prop, actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg.train.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
            optimizer.step()
            if ema is not None:
                ema.update(model)

            loss_val = float(loss.item())
            total_loss += loss_val
            step_end = time.perf_counter()
            compute_s += step_end - step_start
            iter_end = step_end

            if tqdm is not None:
                iterator.set_postfix(loss=f"{(total_loss / step):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if scheduler is not None:
            scheduler.step()

        train_avg = total_loss / max(1, len(loader))

        val_avg: float | None = None
        if val_loader is not None:
            model.eval()
            backup_state = None
            if ema is not None and bool(getattr(cfg.train, "checkpoint_use_ema", True)):
                backup_state = _snapshot_model_state(model)
                ema.copy_to(model)

            val_total = 0.0
            val_steps = 0
            with torch.no_grad():
                for img, prop, actions in val_loader:
                    img = img.to(device)
                    prop = prop.to(device)
                    actions = actions.to(device)
                    val_total += float(model(img, prop, actions).item())
                    val_steps += 1
            val_avg = val_total / max(1, val_steps)

            if backup_state is not None:
                _restore_model_state(model, backup_state)

        selected_metric = float(val_avg if val_avg is not None else train_avg)
        improved = selected_metric < (best_metric - float(cfg.train.save_best_min_delta))
        status = "no_improve"

        if bool(cfg.train.save_best) and improved:
            t_save0 = time.perf_counter()
            backup_state = None
            if ema is not None and bool(getattr(cfg.train, "checkpoint_use_ema", True)):
                backup_state = _snapshot_model_state(model)
                ema.copy_to(model)
            payload = build_checkpoint_payload(
                model=model,
                cfg=cfg,
                epoch=epoch,
                loss=train_avg,
                extra={
                    "best_metric": float(selected_metric),
                    "best_metric_name": best_metric_name,
                    "best_loss": float(selected_metric),
                    "train_loss": float(train_avg),
                    "val_loss": (float(val_avg) if val_avg is not None else None),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "ema": (ema.state_dict() if ema is not None else None),
                },
            )
            save_checkpoint(run_ckpt_dir / "best.pt", payload)
            if backup_state is not None:
                _restore_model_state(model, backup_state)
            save_s = time.perf_counter() - t_save0
            best_metric = selected_metric
            status = "saved best.pt"
        elif selected_metric < best_metric:
            best_metric = selected_metric
            status = "improved(no-save:below-min-delta)"

        dt = time.perf_counter() - epoch_start
        other_s = max(0.0, dt - data_wait_s - compute_s - save_s)
        lr_now = float(optimizer.param_groups[0]["lr"])

        metrics_row = {
            "epoch": int(epoch + 1),
            "loss": float(train_avg),
            "train_loss": float(train_avg),
            "val_loss": (float(val_avg) if val_avg is not None else None),
            "best_metric_name": best_metric_name,
            "best_metric": float(best_metric),
            "best_loss": float(best_metric),
            "lr": float(lr_now),
            "time_s": float(dt),
            "data_s": float(data_wait_s),
            "compute_s": float(compute_s),
            "save_s": float(save_s),
            "other_s": float(other_s),
            "iter_setup_s": float(iter_setup_s),
            "status": status,
        }
        append_jsonl(run_dir / "metrics" / "train_metrics.jsonl", metrics_row)

        print(
            # Print explicit NA when validation is disabled.
            f"Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"train={train_avg:.6f} | "
            f"val={(f'{val_avg:.6f}' if val_avg is not None else 'NA')} | "
            f"best_{best_metric_name}={best_metric:.6f} | "
            f"lr={lr_now:.2e} | time={dt:.1f}s "
            f"(data={data_wait_s:.1f}s, compute={compute_s:.1f}s, save={save_s:.2f}s, "
            f"other={other_s:.1f}s, it_setup={iter_setup_s:.1f}s) | {status}"
        )

    summary = {
        "run_dir": str(run_dir),
        "best_loss": float(best_metric),
        "best_metric": float(best_metric),
        "best_metric_name": best_metric_name,
        "start_epoch": int(start_epoch),
        "epochs": int(epochs),
        "resume_from": str(resume_from) if resume_from else None,
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset) if val_dataset is not None else 0),
        "episodes": int(len(episodes)),
        "best_checkpoint": str(run_ckpt_dir / "best.pt"),
        "action_stats": str(run_stats_path),
    }
    with (run_dir / "metrics" / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Run artifacts saved under: {run_dir}")
    print(f"Best checkpoint        : {run_ckpt_dir / 'best.pt'}")
    return summary
