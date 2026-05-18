from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from mini_pi0.config.io import dump_config
from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys
from mini_pi0.dataset.obs_processor import ObsProcessor
from mini_pi0.eval.core import _episode_seed, evaluate, evaluate_vectorized_maniskill, record_episode, report, save_rollout_grid
from mini_pi0.models.registry import load_checkpoint, make_model
from mini_pi0.sim.registry import make_sim_adapter
from mini_pi0.utils.device import resolve_device
from mini_pi0.utils.parity import build_checkpoint_parity_report, config_diff, format_parity_issues
from mini_pi0.utils.precision import describe_runtime_dtype
from mini_pi0.utils.runs import create_run_dir


def _adapter_factory(cfg: RootConfig):
    """Create per-episode adapter factory with deterministic seed injection.

    Args:
        cfg: Base root config.

    Returns:
        Callable that accepts ``seed`` and returns a simulator adapter.
    """

    def _make(seed: int):
        ep_cfg = copy.deepcopy(cfg)
        ep_cfg.experiment.seed = int(seed)
        return make_sim_adapter(ep_cfg)

    return _make


def _inject_model_cfg_from_checkpoint(cfg: RootConfig, ckpt: dict[str, Any]) -> None:
    """Backfill model config fields from checkpoint metadata.

    Args:
        cfg: Mutable root config to update.
        ckpt: Loaded checkpoint dictionary.
    """

    model_name = ckpt.get("model_name")
    if isinstance(model_name, str) and model_name:
        cfg.model.name = model_name

    model_cfg = ckpt.get("model_config")
    if isinstance(model_cfg, dict):
        for k, v in model_cfg.items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)
        if str(cfg.model.name).strip().lower() == "mini_pi0_fm":
            if "conditioning_mode" not in model_cfg:
                cfg.model.conditioning_mode = "global"
            if "obs_horizon" not in model_cfg:
                cfg.model.obs_horizon = 1
            if "action_attention_causal" not in model_cfg:
                cfg.model.action_attention_causal = False
            is_timm_backbone = str(getattr(cfg.model, "vision_backbone", "")).strip().lower() == "timm"
            if "vision_pretrained" not in model_cfg and is_timm_backbone:
                # Legacy timm FM checkpoints were created with pretrained=False.
                # Avoid downloading unrelated weights before loading their state dict.
                cfg.model.vision_pretrained = False


def _apply_eval_runtime_overrides(cfg: RootConfig) -> None:
    """Apply eval-only simulator overrides that should not affect collection/training."""

    if not bool(getattr(cfg.eval, "disable_domain_randomization", True)):
        return
    env_kwargs = cfg.simulator.env_kwargs
    dr_cfg = env_kwargs.get("domain_randomization")
    if isinstance(dr_cfg, dict):
        dr_cfg["enabled"] = False


def _select_checkpoint_model_state(ckpt: dict[str, Any], weight_source: str) -> dict[str, Any]:
    """Select model weights from a checkpoint payload.

    Args:
        ckpt: Loaded checkpoint dictionary.
        weight_source: One of ``model``, ``raw``, or ``ema``.

    Returns:
        State dict to load into the model.

    Raises:
        ValueError: If the requested weight source is unsupported or unavailable.
    """

    source = str(weight_source or "model").strip().lower()
    if source == "model":
        if "model" in ckpt:
            return ckpt["model"]
        return ckpt

    if source == "raw":
        if "model_raw" in ckpt:
            return ckpt["model_raw"]
        if ckpt.get("model_weight_source") == "raw" and "model" in ckpt:
            return ckpt["model"]
        raise ValueError(
            "Requested eval.weight_source=raw, but checkpoint does not contain raw weights. "
            "Use a checkpoint saved after raw/EMA dual-weight support was added, or evaluate "
            "with eval.weight_source=model."
        )

    if source == "ema":
        ema_state = ckpt.get("ema")
        if isinstance(ema_state, dict) and isinstance(ema_state.get("shadow"), dict):
            return ema_state["shadow"]
        if ckpt.get("model_weight_source") == "ema" and "model" in ckpt:
            return ckpt["model"]
        raise ValueError(
            "Requested eval.weight_source=ema, but checkpoint does not contain EMA weights. "
            "Train with train.ema_decay > 0 or evaluate with eval.weight_source=model."
        )

    raise ValueError("eval.weight_source must be one of: model, raw, ema")


def _resolve_eval_run_dir(cfg: RootConfig) -> Path:
    """Resolve where eval artifacts should be written.

    Preference order:
    1) ``cfg.eval.run_dir`` when explicitly provided.
    2) checkpoint parent run dir when checkpoint path looks like
       ``.../runs/<exp>/runN/checkpoints/best.pt``.
    3) new run dir under ``runs/<experiment>/runN``.
    """

    if getattr(cfg.eval, "run_dir", None):
        run_dir = Path(str(cfg.eval.run_dir))
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        ckpt_path = Path(str(cfg.eval.checkpoint))
        parts = ckpt_path.parts
        run_dir = None
        if "runs" in parts:
            idx = parts.index("runs")
            # runs / <exp> / runN / checkpoints / best.pt
            if len(parts) >= idx + 5 and parts[idx + 3] == "checkpoints":
                run_dir = Path(*parts[: idx + 3])
        if run_dir is None:
            run_dir = create_run_dir(cfg.experiment.runs_root, cfg.experiment.name)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    return run_dir


def _grid_camera_slug(camera: str) -> str:
    """Return a filesystem-safe camera label for grid video artifacts."""

    slug = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(camera).strip())
    return slug.strip("_") or "camera"


def _save_eval_grids(grid_data: dict[str, Any], run_dir: Path, cfg: RootConfig) -> None:
    """Save success/failure rollout grids for each collected eval camera."""

    success_data = grid_data.get("success", {})
    failure_data = grid_data.get("failure", {})
    cameras = list(grid_data.get("cameras") or [])
    if not cameras:
        save_rollout_grid(
            success_data,
            path=str(run_dir / "artifacts" / f"success_grid_{cfg.eval.grid_size}x{cfg.eval.grid_size}.mp4"),
            grid_size=int(cfg.eval.grid_size),
            fps=int(cfg.eval.grid_fps),
        )
        save_rollout_grid(
            failure_data,
            path=str(run_dir / "artifacts" / f"failure_grid_{cfg.eval.grid_size}x{cfg.eval.grid_size}.mp4"),
            grid_size=int(cfg.eval.grid_size),
            fps=int(cfg.eval.grid_fps),
        )
        return

    use_suffix = len(cameras) > 1
    for camera in cameras:
        suffix = f"_{_grid_camera_slug(camera)}" if use_suffix else ""
        save_rollout_grid(
            success_data.get(camera, []),
            path=str(run_dir / "artifacts" / f"success_grid{suffix}_{cfg.eval.grid_size}x{cfg.eval.grid_size}.mp4"),
            grid_size=int(cfg.eval.grid_size),
            fps=int(cfg.eval.grid_fps),
        )
        save_rollout_grid(
            failure_data.get(camera, []),
            path=str(run_dir / "artifacts" / f"failure_grid{suffix}_{cfg.eval.grid_size}x{cfg.eval.grid_size}.mp4"),
            grid_size=int(cfg.eval.grid_size),
            fps=int(cfg.eval.grid_fps),
        )


def run_eval(cfg: RootConfig) -> dict[str, Any]:
    """Run full evaluation pipeline and persist run artifacts.

    Args:
        cfg: Resolved root configuration.

    Returns:
        Summary dictionary containing run directory and metric outputs.
    """

    run_dir = _resolve_eval_run_dir(cfg)
    requested_cfg = copy.deepcopy(cfg)

    device = resolve_device(cfg.eval.device)

    ckpt = load_checkpoint(cfg.eval.checkpoint, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint format is invalid. Expected a dict with 'model' or state_dict keys.")

    parity = build_checkpoint_parity_report(requested_cfg, ckpt)
    parity["checkpoint_path"] = str(cfg.eval.checkpoint)
    strict = bool(getattr(cfg.eval, "strict_parity", True))
    if parity["issues"]:
        msg = "Checkpoint/runtime parity mismatches detected:\n" + format_parity_issues(parity["issues"])
        if strict:
            raise ValueError(msg + "\nDisable with --set eval.strict_parity=false if intentional.")
        print(f"[eval] WARNING: {msg}", flush=True)
    for w in parity.get("warnings", []):
        print(f"[eval] WARNING: {w}", flush=True)

    _inject_model_cfg_from_checkpoint(cfg, ckpt)
    _apply_eval_runtime_overrides(cfg)
    runtime_cfg = copy.deepcopy(cfg)
    print(
        "[eval] Preflight | "
        f"backend={cfg.simulator.backend} task={cfg.simulator.task} robot={cfg.simulator.robot} "
        f"controller={cfg.simulator.controller} obs_mode=image "
        f"image_keys={effective_image_keys(cfg.robot)} "
        f"dtype={describe_runtime_dtype(runtime_dtype=cfg.eval.dtype, model_dtype=None)} "
        f"strict_parity={strict}",
        flush=True,
    )

    dump_config(run_dir / "metrics" / "eval_config_requested.yaml", requested_cfg)
    dump_config(run_dir / "metrics" / "eval_config_runtime.yaml", runtime_cfg)
    with (run_dir / "metrics" / "eval_provenance.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(cfg.eval.checkpoint),
                "strict_parity": strict,
                "parity": parity,
                "requested_to_runtime_diff": config_diff(requested_cfg, runtime_cfg),
            },
            f,
            indent=2,
        )

    model = make_model(cfg).to(device)
    weight_source = str(getattr(cfg.eval, "weight_source", "model"))
    model.load_state_dict(_select_checkpoint_model_state(ckpt, weight_source))
    print(f"[eval] Loaded checkpoint weights | source={weight_source}", flush=True)

    model.eval()

    image_keys = effective_image_keys(cfg.robot)
    stats_path = cfg.eval.action_stats_path or cfg.data.action_stats_path
    state_keys = effective_state_keys(cfg.robot)
    processor = ObsProcessor(
        action_stats_path=stats_path,
        image_key=cfg.robot.image_key,
        image_keys=image_keys,
        proprio_keys=state_keys,
        device=str(device),
        obs_horizon=int(getattr(cfg.model, "obs_horizon", 1)),
        preserve_camera_dim=str(getattr(cfg.model, "conditioning_mode", "global")).strip().lower() == "cross_attention",
    )

    adapter_maker = _adapter_factory(cfg)
    use_vectorized = (
        bool(getattr(cfg.eval, "vectorized", False))
        and str(cfg.simulator.backend).strip().lower() == "maniskill3"
        and not bool(cfg.eval.record)
        and not bool(cfg.eval.record_grid)
    )
    if bool(getattr(cfg.eval, "vectorized", False)) and not use_vectorized:
        print(
            "[eval] Vectorized eval requested but unavailable for this run; "
            "falling back to sequential eval. Vectorized eval requires "
            "backend=maniskill3 and record=false/record_grid=false.",
            flush=True,
        )

    if use_vectorized:
        eval_out = evaluate_vectorized_maniskill(
            model=model,
            processor=processor,
            cfg=cfg,
        )
    else:
        eval_out = evaluate(
            model=model,
            processor=processor,
            cfg=cfg,
            make_adapter=adapter_maker,
            collect_grid=bool(cfg.eval.record_grid),
        )

    if cfg.eval.record_grid and not use_vectorized:
        results, grid_data = eval_out
    else:
        results = eval_out

    plot_path = run_dir / "artifacts" / "eval_metrics.png"
    summary = report(results, plot_path=str(plot_path))

    with (run_dir / "metrics" / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (run_dir / "metrics" / "eval_arrays.json").open("w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in results.items()}, f, indent=2)

    if cfg.eval.record_grid and not use_vectorized:
        _save_eval_grids(grid_data, run_dir, cfg)

    if cfg.eval.record:
        rollout_dir = run_dir / "artifacts" / "rollouts"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        for seed in range(min(5, int(cfg.eval.n_episodes))):
            episode_seed = _episode_seed(cfg, seed)
            record_episode(
                model=model,
                processor=processor,
                cfg=cfg,
                make_adapter=adapter_maker,
                path=str(rollout_dir / f"ep_{seed}.mp4"),
                seed=episode_seed,
            )

    print(f"Run artifacts saved under: {run_dir}")
    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "plot_path": str(plot_path),
    }
