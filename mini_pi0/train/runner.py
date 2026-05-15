from __future__ import annotations

"""Main training entrypoint with simple research-style structure."""

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mini_pi0.config.io import dump_config
from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys
from mini_pi0.dataset.episodes import EpisodeData, load_episodes_from_config
from mini_pi0.dataset.stats import ActionStats
from mini_pi0.dataset.torch_dataset import ActionChunkDataset
from mini_pi0.models.registry import build_checkpoint_payload, load_checkpoint, make_model, save_checkpoint
from mini_pi0.train.augmentation import augment_actions as _augment_actions
from mini_pi0.train.augmentation import augment_image_batch as _augment_image_batch
from mini_pi0.train.data import (
    curate_episodes as _curate_episodes,
)
from mini_pi0.train.data import (
    infer_action_dim,
    infer_prop_dim,
    print_train_header,
    resolve_num_workers,
    seed_everything,
    split_train_val,
    validate_image_observations,
)
from mini_pi0.train.optim import (
    ExponentialMovingAverage,
    build_optimizer as _build_optimizer,
    build_scheduler,
    restore_model_state,
    snapshot_model_state,
)
from mini_pi0.utils.device import resolve_device
from mini_pi0.utils.precision import autocast_context, resolve_runtime_dtype
from mini_pi0.utils.runs import append_jsonl, create_run_dir

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _model_forward_context(cfg: RootConfig, device: torch.device):
    """Create the configured training autocast context."""

    dtype = resolve_runtime_dtype(runtime_dtype=cfg.train.dtype, model_dtype=None)
    return autocast_context(device=device, dtype=dtype)


def _compute_policy_loss(model: torch.nn.Module, cfg: RootConfig, img: torch.Tensor, prop: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Compute model loss with optional FM trajectory regularizers."""

    smooth = float(max(0.0, getattr(cfg.train, "action_smoothness_weight", 0.0)))
    jerk = float(max(0.0, getattr(cfg.train, "action_jerk_weight", 0.0)))
    compute_loss = getattr(model, "compute_loss", None)
    if callable(compute_loss) and (smooth > 0.0 or jerk > 0.0):
        return compute_loss(
            img,
            prop,
            actions,
            smoothness_weight=smooth,
            jerk_weight=jerk,
        )
    return model(img, prop, actions)


def _build_train_checkpoint_payload(
    *,
    model: torch.nn.Module,
    cfg: RootConfig,
    epoch: int,
    train_avg: float,
    val_avg: float | None,
    best_metric: float,
    best_metric_name: str,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    ema: ExponentialMovingAverage | None,
) -> dict[str, Any]:
    """Build a full training checkpoint with optimizer, scheduler, and EMA."""
    payload = build_checkpoint_payload(
        model=model,
        cfg=cfg,
        epoch=epoch,
        loss=train_avg,
        extra={
            "best_metric": float(best_metric),
            "best_metric_name": best_metric_name,
            "best_loss": float(best_metric),
            "train_loss": float(train_avg),
            "val_loss": (float(val_avg) if val_avg is not None else None),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "ema": (ema.state_dict() if ema is not None else None),
        },
    )
    payload["model_weight_source"] = "raw"
    if ema is not None and bool(getattr(cfg.train, "checkpoint_use_ema", True)):
        payload["model_raw"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        payload["model"] = {k: v.detach().cpu() for k, v in ema.shadow.items()}
        payload["model_weight_source"] = "ema"
    return payload


def _run_training_sim_eval(
    *,
    cfg: RootConfig,
    run_dir: Path,
    checkpoint_path: Path,
    epoch: int,
) -> dict[str, Any]:
    """Evaluate a training checkpoint with simulator rollouts."""
    from mini_pi0.eval.runner import run_eval

    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.eval.checkpoint = str(checkpoint_path)
    eval_cfg.eval.action_stats_path = str(run_dir / "artifacts" / "action_stats.json")
    eval_cfg.eval.run_dir = str(run_dir / "sim_eval" / f"epoch_{epoch + 1:03d}")
    eval_cfg.eval.n_episodes = int(max(1, getattr(cfg.train, "sim_eval_n_episodes", 10)))
    sim_eval_max_steps = getattr(cfg.train, "sim_eval_max_steps", None)
    eval_cfg.eval.max_steps = int(sim_eval_max_steps) if sim_eval_max_steps is not None else cfg.eval.max_steps
    eval_cfg.eval.record_grid = bool(getattr(cfg.train, "sim_eval_record_grid", False))
    eval_cfg.eval.record = False
    eval_cfg.eval.device = cfg.train.device
    return run_eval(eval_cfg)


def _prepare_episodes_and_dataset(cfg: RootConfig, run_dir: Path) -> tuple[list[EpisodeData], dict[str, Any], Any, Any | None, Path]:
    """Load data, align config dims, build action-chunk dataset, and split train/val."""
    print(
        "[train] Loading dataset | "
        f"format={cfg.data.format} n_demos={cfg.data.n_demos}",
        flush=True,
    )
    episodes = load_episodes_from_config(cfg)
    print(f"[train] Dataset loaded | episodes={len(episodes)}", flush=True)

    episodes, curation_summary = _curate_episodes(episodes, cfg)
    if curation_summary.get("enabled", False):
        print(
            "[train] Data curation | "
            f"before={curation_summary['before_episodes']} after={curation_summary['after_episodes']} "
            f"removed={curation_summary['removed_episodes']} "
            f"progress_key={curation_summary.get('progress_key')}",
            flush=True,
        )
        if curation_summary.get("reasons"):
            print(f"[train] Data curation reasons | {curation_summary['reasons']}", flush=True)

    state_keys = effective_state_keys(cfg.robot)
    image_keys = effective_image_keys(cfg.robot)

    inferred_action_dim = infer_action_dim(episodes)
    inferred_prop_dim = infer_prop_dim(episodes[0].obs[0], state_keys)
    validate_image_observations(episodes[0].obs[0], image_keys)
    if cfg.robot.action_dim != inferred_action_dim:
        print(
            f"[train] Overriding robot.action_dim from {cfg.robot.action_dim} to inferred {inferred_action_dim} "
            "based on dataset actions."
        )
    if cfg.model.action_dim != inferred_action_dim:
        print(
            f"[train] Overriding model.action_dim from {cfg.model.action_dim} to inferred {inferred_action_dim} "
            "based on dataset actions."
        )
    if cfg.model.prop_dim != inferred_prop_dim:
        print(
            f"[train] Overriding model.prop_dim from {cfg.model.prop_dim} to inferred {inferred_prop_dim} "
            "based on dataset state keys."
        )
    cfg.robot.action_dim = inferred_action_dim
    cfg.model.action_dim = inferred_action_dim
    cfg.model.prop_dim = inferred_prop_dim
    dump_config(run_dir / "config_resolved.yaml", cfg)

    all_actions = np.concatenate([ep.actions.astype(np.float32) for ep in episodes], axis=0)
    stats = ActionStats.from_actions(all_actions)
    run_stats_path = run_dir / "artifacts" / "action_stats.json"
    stats.save(str(run_stats_path))

    dataset = ActionChunkDataset(
        episodes=episodes,
        chunk_size=cfg.data.chunk_size,
        image_key=cfg.robot.image_key,
        image_keys=image_keys,
        proprio_keys=state_keys,
        action_stats=stats,
        obs_horizon=int(getattr(cfg.model, "obs_horizon", 1)),
        preserve_camera_dim=str(getattr(cfg.model, "conditioning_mode", "global")).strip().lower() == "cross_attention",
    )
    train_dataset, val_dataset = split_train_val(
        dataset,
        val_ratio=float(getattr(cfg.train, "val_ratio", 0.0)),
        seed=int(cfg.experiment.seed),
    )
    print(
        f"[train] Prepared action-chunk dataset | total={len(dataset)} train={len(train_dataset)} "
        f"val={(len(val_dataset) if val_dataset is not None else 0)}",
        flush=True,
    )
    return episodes, curation_summary, train_dataset, val_dataset, run_stats_path


def _build_dataloaders(
    train_dataset: Any,
    val_dataset: Any | None,
    cfg: RootConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader | None, int, bool]:
    """Create train and optional validation dataloaders."""
    num_workers = resolve_num_workers(cfg.train.num_workers)
    use_persistent = bool(cfg.train.persistent_workers and num_workers > 0)
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(cfg.train.batch_size),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
        "drop_last": False,
        "persistent_workers": use_persistent,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = None
    if val_dataset is not None:
        val_kwargs = dict(loader_kwargs)
        val_kwargs["shuffle"] = False
        val_loader = DataLoader(val_dataset, **val_kwargs)
    return loader, val_loader, num_workers, use_persistent


def _restore_from_checkpoint(
    cfg: RootConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ema: ExponentialMovingAverage | None,
    best_metric_name: str,
) -> tuple[int, float, str | None]:
    """Restore model/optimizer/scheduler/ema state when resume checkpoint is provided."""
    resume_from = getattr(cfg.train, "resume_from", None)
    if not resume_from:
        return 0, float("inf"), None

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
    return start_epoch, best_metric, str(resume_from)


def run_train(cfg: RootConfig) -> dict[str, Any]:
    """Execute end-to-end supervised training for the configured action model."""
    run_dir = create_run_dir(cfg.experiment.runs_root, cfg.experiment.name)
    dump_config(run_dir / "config_resolved.yaml", cfg)
    seed_everything(int(cfg.experiment.seed))

    episodes, curation_summary, train_dataset, val_dataset, run_stats_path = _prepare_episodes_and_dataset(cfg, run_dir)

    model = make_model(cfg)
    device = resolve_device(cfg.train.device)
    model = model.to(device)
    print_train_header(cfg, device, n_episodes=len(episodes), n_samples=len(train_dataset), model=model)

    loader, val_loader, num_workers, use_persistent = _build_dataloaders(train_dataset, val_dataset, cfg, device)

    optimizer, lr_summary = _build_optimizer(model, cfg)
    scheduler, scheduler_desc = build_scheduler(optimizer, cfg)
    ema_decay = float(getattr(cfg.train, "ema_decay", 0.0))
    ema = ExponentialMovingAverage(model, decay=ema_decay) if ema_decay > 0.0 else None
    best_metric_name = "val_loss" if val_loader is not None else "train_loss"
    start_epoch, best_metric, resume_from = _restore_from_checkpoint(
        cfg,
        model,
        optimizer,
        scheduler,
        ema,
        best_metric_name,
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
    print(
        "Optimizer          : AdamW("
        f"backbone_lr={lr_summary['backbone_lr']:.2e}, "
        f"expert_lr={lr_summary['expert_lr']:.2e}, "
        f"weight_decay={cfg.train.weight_decay})"
    )
    print(f"Scheduler          : {scheduler_desc}")
    print(
        "Augmentation       : "
        f"image_enable={bool(getattr(cfg.train, 'image_aug_enable', False))}, "
        f"crop_scale={float(getattr(cfg.train, 'image_aug_crop_scale', 1.0)):.3f}, "
        f"brightness={float(getattr(cfg.train, 'image_aug_brightness', 0.0)):.3f}, "
        f"contrast={float(getattr(cfg.train, 'image_aug_contrast', 0.0)):.3f}, "
        f"saturation={float(getattr(cfg.train, 'image_aug_saturation', 0.0)):.3f}, "
        f"action_noise_std={float(getattr(cfg.train, 'action_noise_std', 0.0)):.4f}, "
        f"action_noise_clip={float(getattr(cfg.train, 'action_noise_clip', 0.0)):.3f}"
    )
    print(
        "EMA                : "
        f"enabled={ema is not None}, decay={float(getattr(cfg.train, 'ema_decay', 0.0)):.6f}, "
        f"checkpoint_use_ema={bool(getattr(cfg.train, 'checkpoint_use_ema', True))}, "
        f"val_use_ema={bool(getattr(cfg.train, 'val_use_ema', False))}"
    )
    sim_eval_every = int(max(0, getattr(cfg.train, "sim_eval_every_epochs", 0)))
    print(
        "Sim eval           : "
        f"enabled={sim_eval_every > 0}, every_epochs={sim_eval_every}, "
        f"n_episodes={int(getattr(cfg.train, 'sim_eval_n_episodes', 10))}, "
        f"max_steps={getattr(cfg.train, 'sim_eval_max_steps', None)}, "
        f"record_grid={bool(getattr(cfg.train, 'sim_eval_record_grid', False))}, "
        f"save_best_success={bool(getattr(cfg.train, 'save_best_success', True))}"
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

    global_step = 0
    use_ema_for_val = bool(getattr(cfg.train, "val_use_ema", False))
    best_success_rate = float("-inf")
    best_success_epoch: int | None = None

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
            img = _augment_image_batch(img, cfg)
            actions = _augment_actions(actions, cfg)

            with _model_forward_context(cfg, device):
                loss = _compute_policy_loss(model, cfg, img, prop, actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg.train.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
            optimizer.step()
            if ema is not None:
                ema.update(model)
            global_step += 1

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
            if ema is not None and use_ema_for_val:
                backup_state = snapshot_model_state(model)
                ema.copy_to(model)
            val_total = 0.0
            val_steps = 0
            with torch.no_grad():
                for img, prop, actions in val_loader:
                    img = img.to(device)
                    prop = prop.to(device)
                    actions = actions.to(device)
                    with _model_forward_context(cfg, device):
                        val_total += float(_compute_policy_loss(model, cfg, img, prop, actions).item())
                    val_steps += 1
            val_avg = val_total / max(1, val_steps)
            if backup_state is not None:
                restore_model_state(model, backup_state)

        selected_metric = float(val_avg if val_avg is not None else train_avg)
        improved = selected_metric < (best_metric - float(cfg.train.save_best_min_delta))
        status = "no_improve"
        if bool(cfg.train.save_best) and improved:
            t_save0 = time.perf_counter()
            payload = _build_train_checkpoint_payload(
                model=model,
                cfg=cfg,
                epoch=epoch,
                train_avg=train_avg,
                val_avg=val_avg,
                best_metric=selected_metric,
                best_metric_name=best_metric_name,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
            )
            save_checkpoint(run_ckpt_dir / "best.pt", payload)
            save_s = time.perf_counter() - t_save0
            best_metric = selected_metric
            status = "saved best.pt"
        elif selected_metric < best_metric:
            best_metric = selected_metric
            status = "improved(no-save:below-min-delta)"

        sim_eval_summary: dict[str, Any] | None = None
        if sim_eval_every > 0 and (epoch + 1) % sim_eval_every == 0:
            t_eval0 = time.perf_counter()
            latest_payload = _build_train_checkpoint_payload(
                model=model,
                cfg=cfg,
                epoch=epoch,
                train_avg=train_avg,
                val_avg=val_avg,
                best_metric=best_metric,
                best_metric_name=best_metric_name,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
            )
            latest_path = run_ckpt_dir / "latest.pt"
            save_checkpoint(latest_path, latest_payload)

            eval_result = _run_training_sim_eval(
                cfg=cfg,
                run_dir=run_dir,
                checkpoint_path=latest_path,
                epoch=epoch,
            )
            sim_eval_summary = dict(eval_result.get("summary", {}))
            sim_success = float(sim_eval_summary.get("success_rate", 0.0))
            sim_eval_summary.update(
                {
                    "epoch": int(epoch + 1),
                    "checkpoint": str(latest_path),
                    "eval_run_dir": str(eval_result.get("run_dir", "")),
                    "elapsed_s": float(time.perf_counter() - t_eval0),
                }
            )
            append_jsonl(run_dir / "metrics" / "sim_eval_metrics.jsonl", sim_eval_summary)
            min_success_to_save = float(getattr(cfg.train, "save_best_success_min_rate", 0.0))
            if (
                bool(getattr(cfg.train, "save_best_success", True))
                and sim_success > best_success_rate
                and sim_success > min_success_to_save
            ):
                best_success_rate = sim_success
                best_success_epoch = int(epoch + 1)
                save_checkpoint(
                    run_ckpt_dir / "best_success.pt",
                    {
                        **latest_payload,
                        "best_success_rate": float(best_success_rate),
                        "best_success_epoch": int(best_success_epoch),
                        "sim_eval_summary": sim_eval_summary,
                    },
                )
                status = f"{status}; saved best_success.pt"
            else:
                status = f"{status}; sim_eval_success={sim_success:.3f}"

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
        if sim_eval_summary is not None:
            metrics_row["sim_success_rate"] = float(sim_eval_summary.get("success_rate", 0.0))
            metrics_row["sim_episode_len_mean"] = float(sim_eval_summary.get("episode_len_mean", 0.0))
            metrics_row["sim_action_clip_fraction_mean"] = float(
                sim_eval_summary.get("action_clip_fraction_mean", 0.0)
            )
        append_jsonl(run_dir / "metrics" / "train_metrics.jsonl", metrics_row)
        print(
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
        "data_curation": curation_summary,
        "best_checkpoint": str(run_ckpt_dir / "best.pt"),
        "best_success_checkpoint": str(run_ckpt_dir / "best_success.pt"),
        "best_success_rate": (float(best_success_rate) if best_success_epoch is not None else None),
        "best_success_epoch": best_success_epoch,
        "action_stats": str(run_stats_path),
    }
    with (run_dir / "metrics" / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Run artifacts saved under: {run_dir}")
    print(f"Best checkpoint        : {run_ckpt_dir / 'best.pt'}")
    if best_success_epoch is not None:
        print(
            f"Best success checkpoint: {run_ckpt_dir / 'best_success.pt'} "
            f"(success_rate={best_success_rate:.3f}, epoch={best_success_epoch})"
        )
    return summary


__all__ = [
    "run_train",
    "_curate_episodes",
    "_build_optimizer",
    "_augment_image_batch",
    "_augment_actions",
]
