from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mini_pi0.config.io import dump_config
from mini_pi0.config.schema import RootConfig, effective_state_keys, to_dict
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


def _infer_visual_mode_and_dim(obs: dict[str, np.ndarray], observation_key: str) -> tuple[str, int]:
    """Infer visual input mode and dimension from one observation sample."""

    visual = np.asarray(obs[observation_key])
    if visual.ndim == 1:
        return "feature", int(visual.shape[0])
    if visual.ndim == 3:
        return "image", 0
    raise ValueError(
        f"Unsupported visual observation shape {visual.shape} for key '{observation_key}'. "
        "Expected feature [D] or image [H,W,C]."
    )


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
    obs_mode_cfg = str(getattr(cfg.data, "observation_mode", "image")).strip().lower()
    observation_key = (
        cfg.data.precomputed_feature_key if obs_mode_cfg in {"precomputed", "feature", "features"} else cfg.robot.image_key
    )

    inferred_action_dim = _infer_action_dim(episodes)
    inferred_prop_dim = _infer_prop_dim(episodes[0].obs[0], state_keys)
    inferred_obs_mode, inferred_vision_dim = _infer_visual_mode_and_dim(episodes[0].obs[0], observation_key)

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
            f"based on observation key '{observation_key}'."
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

    run_stats_path = run_dir / "artifacts" / "action_stats.json"
    stats.save(str(run_stats_path))

    dataset = ActionChunkDataset(
        episodes=episodes,
        chunk_size=cfg.data.chunk_size,
        image_key=cfg.robot.image_key,
        proprio_keys=state_keys,
        action_stats=stats,
        observation_key=observation_key,
    )
    print(f"[train] Prepared action-chunk dataset | samples={len(dataset)}", flush=True)

    model = make_model(cfg)
    device = resolve_device(cfg.train.device)
    model = model.to(device)

    _print_header(cfg, device, n_episodes=len(episodes), n_samples=len(dataset), model=model)

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

    loader = DataLoader(dataset, **loader_kwargs)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler, scheduler_desc = _build_scheduler(optimizer, cfg)

    start_epoch = 0
    best_loss = float("inf")
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
        best_loss = float(ckpt.get("best_loss", ckpt.get("loss", float("inf"))))

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

        print(
            f"[train] Resume state | start_epoch={start_epoch + 1} "
            f"previous_epoch={prev_epoch + 1} best_loss={best_loss:.6f}",
            flush=True,
        )

    print(f"Training on device : {device}")
    print(
        f"Dataloader         : batch_size={cfg.train.batch_size}, num_workers={num_workers}, "
        f"pin_memory={device.type == 'cuda'}, persistent_workers={use_persistent}"
    )
    print(f"Optimizer          : AdamW(lr={cfg.train.lr}, weight_decay={cfg.train.weight_decay})")
    print(f"Scheduler          : {scheduler_desc}")
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

            loss_val = float(loss.item())
            total_loss += loss_val
            step_end = time.perf_counter()
            compute_s += step_end - step_start
            iter_end = step_end

            if tqdm is not None:
                iterator.set_postfix(loss=f"{(total_loss / step):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if scheduler is not None:
            scheduler.step()

        avg = total_loss / max(1, len(loader))
        improved = avg < (best_loss - float(cfg.train.save_best_min_delta))
        status = "no_improve"

        if bool(cfg.train.save_best) and improved:
            t_save0 = time.perf_counter()
            payload = build_checkpoint_payload(
                model=model,
                cfg=cfg,
                epoch=epoch,
                loss=avg,
                extra={
                    "best_loss": float(avg),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                },
            )
            save_checkpoint(run_ckpt_dir / "best.pt", payload)
            save_s = time.perf_counter() - t_save0
            best_loss = avg
            status = "saved best.pt"
        elif avg < best_loss:
            best_loss = avg
            status = "improved(no-save:below-min-delta)"

        dt = time.perf_counter() - epoch_start
        other_s = max(0.0, dt - data_wait_s - compute_s - save_s)
        lr_now = float(optimizer.param_groups[0]["lr"])

        metrics_row = {
            "epoch": int(epoch + 1),
            "loss": float(avg),
            "best_loss": float(best_loss),
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
            f"Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"loss={avg:.6f} | best={best_loss:.6f} | "
            f"lr={lr_now:.2e} | time={dt:.1f}s "
            f"(data={data_wait_s:.1f}s, compute={compute_s:.1f}s, save={save_s:.2f}s, "
            f"other={other_s:.1f}s, it_setup={iter_setup_s:.1f}s) | {status}"
        )

    summary = {
        "run_dir": str(run_dir),
        "best_loss": float(best_loss),
        "start_epoch": int(start_epoch),
        "epochs": int(epochs),
        "resume_from": str(resume_from) if resume_from else None,
        "train_samples": int(len(dataset)),
        "episodes": int(len(episodes)),
        "best_checkpoint": str(run_ckpt_dir / "best.pt"),
        "action_stats": str(run_stats_path),
    }
    with (run_dir / "metrics" / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Run artifacts saved under: {run_dir}")
    print(f"Best checkpoint        : {run_ckpt_dir / 'best.pt'}")
    return summary
