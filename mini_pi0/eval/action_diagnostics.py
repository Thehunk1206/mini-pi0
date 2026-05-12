from __future__ import annotations

"""Offline action diagnostics for flow-matching policy checkpoints.

This module compares sampled model action chunks against dataset action chunks
without running closed-loop simulator rollouts. It is intended for diagnosing
which action dimensions are causing out-of-bounds clipping during eval.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mini_pi0.config.io import load_config
from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys
from mini_pi0.dataset.episodes import load_episodes_from_config
from mini_pi0.dataset.stats import ActionStats
from mini_pi0.dataset.torch_dataset import ActionChunkDataset
from mini_pi0.eval.runner import (
    _inject_model_cfg_from_checkpoint,
    _select_checkpoint_model_state,
)
from mini_pi0.models.registry import load_checkpoint, make_model
from mini_pi0.sim.registry import make_sim_adapter
from mini_pi0.train.data import curate_episodes
from mini_pi0.utils.device import resolve_device
from mini_pi0.utils.precision import autocast_context, resolve_runtime_dtype


def _build_dataset(cfg: RootConfig, stats: ActionStats) -> ActionChunkDataset:
    """Build the action chunk dataset used for offline comparison."""

    episodes = load_episodes_from_config(cfg)
    episodes, _ = curate_episodes(episodes, cfg)
    obs_mode_cfg = str(getattr(cfg.data, "observation_mode", "image")).strip().lower()
    observation_key = cfg.data.precomputed_feature_key if obs_mode_cfg in {"precomputed", "feature", "features"} else None
    return ActionChunkDataset(
        episodes=episodes,
        chunk_size=int(cfg.data.chunk_size),
        image_key=cfg.robot.image_key,
        image_keys=effective_image_keys(cfg.robot),
        proprio_keys=effective_state_keys(cfg.robot),
        action_stats=stats,
        observation_key=observation_key,
        obs_horizon=int(getattr(cfg.model, "obs_horizon", 1)),
        preserve_camera_dim=str(getattr(cfg.model, "conditioning_mode", "global")).strip().lower() == "cross_attention",
    )


def _select_subset(dataset: ActionChunkDataset, num_samples: int, seed: int) -> Subset:
    """Select a deterministic subset without materializing the full dataset."""

    n = len(dataset)
    if n == 0:
        raise ValueError("Action diagnostic dataset is empty after filtering.")
    take = min(int(max(1, num_samples)), n)
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(n, size=take, replace=False)
    return Subset(dataset, indices.tolist())


def _action_bounds(cfg: RootConfig) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Best-effort simulator action bounds lookup."""

    adapter = make_sim_adapter(cfg)
    try:
        low, high = adapter.action_spec()
        return np.asarray(low, dtype=np.float32).reshape(-1), np.asarray(high, dtype=np.float32).reshape(-1)
    finally:
        adapter.close()


def _load_model(cfg: RootConfig, checkpoint: str, weight_source: str, device: torch.device) -> torch.nn.Module:
    """Instantiate a model and load the requested checkpoint weights."""

    ckpt = load_checkpoint(checkpoint, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint format is invalid. Expected a dict with model weights.")
    _inject_model_cfg_from_checkpoint(cfg, ckpt)
    model = make_model(cfg).to(device)
    model.load_state_dict(_select_checkpoint_model_state(ckpt, weight_source))
    model.eval()
    return model


def _empty_metric_sums(action_dim: int) -> dict[str, np.ndarray | float | int]:
    """Create accumulation buffers for one diagnostic run."""

    zeros = np.zeros((action_dim,), dtype=np.float64)
    return {
        "count": 0,
        "action_count": 0,
        "norm_abs_error": zeros.copy(),
        "raw_abs_error": zeros.copy(),
        "pred_raw_sum": zeros.copy(),
        "pred_raw_sq_sum": zeros.copy(),
        "gt_raw_sum": zeros.copy(),
        "gt_raw_sq_sum": zeros.copy(),
        "pred_raw_abs_max": zeros.copy(),
        "gt_raw_abs_max": zeros.copy(),
        "pred_clip_low": zeros.copy(),
        "pred_clip_high": zeros.copy(),
        "gt_oob_low": zeros.copy(),
        "gt_oob_high": zeros.copy(),
        "any_pred_clip": 0,
        "any_gt_oob": 0,
    }


def _update_metrics(
    metrics: dict[str, np.ndarray | float | int],
    pred_norm: np.ndarray,
    gt_norm: np.ndarray,
    pred_raw: np.ndarray,
    gt_raw: np.ndarray,
    low: np.ndarray | None,
    high: np.ndarray | None,
) -> None:
    """Accumulate per-dimension prediction and bounds metrics."""

    pred_norm_flat = pred_norm.reshape(-1, pred_norm.shape[-1])
    gt_norm_flat = gt_norm.reshape(-1, gt_norm.shape[-1])
    pred_raw_flat = pred_raw.reshape(-1, pred_raw.shape[-1])
    gt_raw_flat = gt_raw.reshape(-1, gt_raw.shape[-1])
    n = int(pred_norm_flat.shape[0])
    metrics["count"] = int(metrics["count"]) + n
    metrics["action_count"] = int(metrics["action_count"]) + n
    metrics["norm_abs_error"] += np.abs(pred_norm_flat - gt_norm_flat).sum(axis=0)
    metrics["raw_abs_error"] += np.abs(pred_raw_flat - gt_raw_flat).sum(axis=0)
    metrics["pred_raw_sum"] += pred_raw_flat.sum(axis=0)
    metrics["pred_raw_sq_sum"] += np.square(pred_raw_flat).sum(axis=0)
    metrics["gt_raw_sum"] += gt_raw_flat.sum(axis=0)
    metrics["gt_raw_sq_sum"] += np.square(gt_raw_flat).sum(axis=0)
    metrics["pred_raw_abs_max"] = np.maximum(metrics["pred_raw_abs_max"], np.abs(pred_raw_flat).max(axis=0))
    metrics["gt_raw_abs_max"] = np.maximum(metrics["gt_raw_abs_max"], np.abs(gt_raw_flat).max(axis=0))
    if low is None or high is None:
        return
    pred_low = pred_raw_flat < low
    pred_high = pred_raw_flat > high
    gt_low = gt_raw_flat < low
    gt_high = gt_raw_flat > high
    metrics["pred_clip_low"] += pred_low.sum(axis=0)
    metrics["pred_clip_high"] += pred_high.sum(axis=0)
    metrics["gt_oob_low"] += gt_low.sum(axis=0)
    metrics["gt_oob_high"] += gt_high.sum(axis=0)
    metrics["any_pred_clip"] = int(metrics["any_pred_clip"]) + int(np.any(pred_low | pred_high, axis=1).sum())
    metrics["any_gt_oob"] = int(metrics["any_gt_oob"]) + int(np.any(gt_low | gt_high, axis=1).sum())


def _finalize_metrics(metrics: dict[str, np.ndarray | float | int]) -> dict[str, Any]:
    """Convert accumulated sums into JSON-serializable means/fractions."""

    n = max(1, int(metrics["count"]))
    pred_mean = metrics["pred_raw_sum"] / n
    gt_mean = metrics["gt_raw_sum"] / n
    pred_var = np.maximum(metrics["pred_raw_sq_sum"] / n - np.square(pred_mean), 0.0)
    gt_var = np.maximum(metrics["gt_raw_sq_sum"] / n - np.square(gt_mean), 0.0)
    pred_clip = (metrics["pred_clip_low"] + metrics["pred_clip_high"]) / n
    gt_oob = (metrics["gt_oob_low"] + metrics["gt_oob_high"]) / n
    return {
        "num_actions": int(metrics["action_count"]),
        "normalized_mae_by_dim": (metrics["norm_abs_error"] / n).tolist(),
        "raw_mae_by_dim": (metrics["raw_abs_error"] / n).tolist(),
        "pred_raw_mean_by_dim": pred_mean.tolist(),
        "pred_raw_std_by_dim": np.sqrt(pred_var).tolist(),
        "pred_raw_abs_max_by_dim": metrics["pred_raw_abs_max"].tolist(),
        "gt_raw_mean_by_dim": gt_mean.tolist(),
        "gt_raw_std_by_dim": np.sqrt(gt_var).tolist(),
        "gt_raw_abs_max_by_dim": metrics["gt_raw_abs_max"].tolist(),
        "pred_clip_fraction_by_dim": pred_clip.tolist(),
        "pred_clip_low_fraction_by_dim": (metrics["pred_clip_low"] / n).tolist(),
        "pred_clip_high_fraction_by_dim": (metrics["pred_clip_high"] / n).tolist(),
        "pred_any_clip_fraction": float(int(metrics["any_pred_clip"]) / n),
        "gt_oob_fraction_by_dim": gt_oob.tolist(),
        "gt_any_oob_fraction": float(int(metrics["any_gt_oob"]) / n),
    }


def run_action_diagnostics(
    cfg: RootConfig,
    *,
    checkpoint: str,
    action_stats_path: str,
    flow_steps: Sequence[int],
    num_samples: int,
    batch_size: int,
    weight_source: str,
    device_name: str,
    output_json: str | None,
) -> dict[str, Any]:
    """Compare model-sampled action chunks with dataset ground-truth chunks."""

    device = resolve_device(device_name)
    stats = ActionStats.load(action_stats_path)
    cfg = copy.deepcopy(cfg)
    cfg.eval.checkpoint = checkpoint
    cfg.eval.action_stats_path = action_stats_path
    model = _load_model(cfg, checkpoint=checkpoint, weight_source=weight_source, device=device)
    dataset = _build_dataset(cfg, stats)
    subset = _select_subset(dataset, num_samples=num_samples, seed=int(cfg.experiment.seed))
    loader = DataLoader(subset, batch_size=int(max(1, batch_size)), shuffle=False, num_workers=0)
    low, high = _action_bounds(cfg)
    action_dim = int(cfg.model.action_dim)
    results: dict[str, Any] = {
        "checkpoint": checkpoint,
        "action_stats_path": action_stats_path,
        "num_samples": len(subset),
        "chunk_size": int(cfg.model.chunk_size),
        "action_dim": action_dim,
        "action_low": low.tolist() if low is not None else None,
        "action_high": high.tolist() if high is not None else None,
        "flow_steps": {},
    }
    dtype = resolve_runtime_dtype(runtime_dtype=cfg.eval.dtype, model_dtype=cfg.model.dtype)
    for steps in flow_steps:
        metrics = _empty_metric_sums(action_dim)
        with torch.no_grad(), autocast_context(device=device, dtype=dtype):
            for img, prop, gt_norm_t in loader:
                img = img.to(device=device)
                prop = prop.to(device=device)
                gt_norm_t = gt_norm_t.to(device=device)
                pred_norm_t = model.sample(
                    img,
                    prop,
                    n_steps=int(max(1, steps)),
                    solver=str(getattr(cfg.eval, "flow_solver", "euler")),
                )
                pred_norm = pred_norm_t.detach().float().cpu().numpy()
                gt_norm = gt_norm_t.detach().float().cpu().numpy()
                pred_raw = stats.denormalize(pred_norm)
                gt_raw = stats.denormalize(gt_norm)
                _update_metrics(metrics, pred_norm, gt_norm, pred_raw, gt_raw, low, high)
        results["flow_steps"][str(int(steps))] = _finalize_metrics(metrics)

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _print_summary(results)
    return results


def _format_dim_values(values: Sequence[float], scale: float = 1.0) -> str:
    """Format per-dimension values compactly."""

    return ", ".join(f"d{i}={scale * float(v):.3g}" for i, v in enumerate(values))


def _print_summary(results: dict[str, Any]) -> None:
    """Print a compact human-readable diagnostic summary."""

    print(
        "[action-diagnostics] "
        f"samples={results['num_samples']} chunk={results['chunk_size']} action_dim={results['action_dim']}"
    )
    if results.get("action_low") is not None:
        print(f"[action-diagnostics] low : {_format_dim_values(results['action_low'])}")
        print(f"[action-diagnostics] high: {_format_dim_values(results['action_high'])}")
    for steps, row in results["flow_steps"].items():
        print(f"\nflow_steps={steps}")
        print(f"  pred any clip : {100.0 * float(row['pred_any_clip_fraction']):.1f}%")
        print(f"  gt any oob    : {100.0 * float(row['gt_any_oob_fraction']):.1f}%")
        print(f"  pred clip dim : {_format_dim_values(row['pred_clip_fraction_by_dim'], scale=100.0)}")
        print(f"  raw MAE dim   : {_format_dim_values(row['raw_mae_by_dim'])}")
        print(f"  norm MAE dim  : {_format_dim_values(row['normalized_mae_by_dim'])}")
        print(f"  pred abs max  : {_format_dim_values(row['pred_raw_abs_max_by_dim'])}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for module execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate offline.")
    parser.add_argument("--action_stats", required=True, help="Action stats JSON used by the checkpoint/run.")
    parser.add_argument("--flow_steps", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_source", choices=["model", "raw", "ema"], default="raw")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> int:
    """Run action diagnostics from the command line."""

    args = _parse_args()
    cfg = load_config(args.config, overrides=list(args.overrides or []))
    run_action_diagnostics(
        cfg,
        checkpoint=str(args.checkpoint),
        action_stats_path=str(args.action_stats),
        flow_steps=list(args.flow_steps),
        num_samples=int(args.num_samples),
        batch_size=int(args.batch_size),
        weight_source=str(args.weight_source),
        device_name=str(args.device),
        output_json=args.output_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
