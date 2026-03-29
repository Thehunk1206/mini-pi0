from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from mini_pi0.config.io import dump_config
from mini_pi0.config.schema import RootConfig, effective_state_keys
from mini_pi0.dataset.obs_processor import ObsProcessor
from mini_pi0.eval.core import evaluate, record_episode, report, save_rollout_grid
from mini_pi0.models.registry import load_checkpoint, make_model
from mini_pi0.sim.registry import make_sim_adapter
from mini_pi0.utils.device import resolve_device
from mini_pi0.utils.runs import create_run_dir
from mini_pi0.vision.encoders import build_vision_extractor


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

    # Keep runtime vision config under user control (config / CLI overrides).
    # Checkpoint vision metadata can be stale relative to the model's expected
    # feature dimension when users switch encoders intentionally.
    vision_cfg = ckpt.get("vision_config")
    if isinstance(vision_cfg, dict):
        for k, v in vision_cfg.items():
            if not hasattr(cfg.vision, k):
                continue
            cur = getattr(cfg.vision, k)
            if cur is None or (isinstance(cur, str) and not cur.strip()):
                setattr(cfg.vision, k, v)


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


def run_eval(cfg: RootConfig) -> dict[str, Any]:
    """Run full evaluation pipeline and persist run artifacts.

    Args:
        cfg: Resolved root configuration.

    Returns:
        Summary dictionary containing run directory and metric outputs.
    """

    run_dir = _resolve_eval_run_dir(cfg)
    dump_config(run_dir / "config_resolved.yaml", cfg)

    device = resolve_device(cfg.eval.device)

    ckpt = load_checkpoint(cfg.eval.checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        _inject_model_cfg_from_checkpoint(cfg, ckpt)

    model = make_model(cfg).to(device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        raise ValueError("Checkpoint format is invalid. Expected a dict with 'model' or state_dict keys.")

    model.eval()

    feature_extractor = None
    if str(cfg.model.obs_mode).strip().lower() in {"feature", "precomputed", "features"}:
        if bool(cfg.vision.use_runtime_extractor):
            feature_extractor = build_vision_extractor(
                backend=cfg.vision.backend,
                model_name=cfg.vision.model_name,
                pretrained=bool(cfg.vision.pretrained),
                image_size=int(cfg.vision.image_size),
                hf_model_id=cfg.vision.hf_model_id,
                local_files_only=bool(cfg.vision.local_files_only),
                device=device,
            )
            expected_dim = int(cfg.model.vision_dim)
            got_dim = int(feature_extractor.feature_dim)
            if expected_dim > 0 and got_dim != expected_dim:
                raise ValueError(
                    "Vision feature dim mismatch: model expects "
                    f"{expected_dim}, runtime extractor '{cfg.vision.model_name}' outputs {got_dim}. "
                    "Set a matching encoder via `--set vision.model_name=...` "
                    "or use a checkpoint/config trained with the same feature backend."
                )
        else:
            raise ValueError(
                "Model expects feature observations but vision.use_runtime_extractor=false. "
                "Enable runtime extractor for eval/deploy."
            )

    stats_path = cfg.eval.action_stats_path or cfg.data.action_stats_path
    state_keys = effective_state_keys(cfg.robot)
    processor = ObsProcessor(
        action_stats_path=stats_path,
        image_key=cfg.robot.image_key,
        proprio_keys=state_keys,
        device=str(device),
        observation_mode=cfg.model.obs_mode,
        feature_key=cfg.data.precomputed_feature_key,
        feature_extractor=feature_extractor,
    )

    adapter_maker = _adapter_factory(cfg)
    eval_out = evaluate(
        model=model,
        processor=processor,
        cfg=cfg,
        make_adapter=adapter_maker,
        collect_grid=bool(cfg.eval.record_grid),
    )

    if cfg.eval.record_grid:
        results, grid_data = eval_out
    else:
        results = eval_out

    plot_path = run_dir / "artifacts" / "eval_metrics.png"
    summary = report(results, plot_path=str(plot_path))

    with (run_dir / "metrics" / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (run_dir / "metrics" / "eval_arrays.json").open("w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in results.items()}, f, indent=2)

    if cfg.eval.record_grid:
        save_rollout_grid(
            grid_data["success"],
            path=str(run_dir / "artifacts" / f"success_grid_{cfg.eval.grid_size}x{cfg.eval.grid_size}.mp4"),
            grid_size=int(cfg.eval.grid_size),
            fps=int(cfg.eval.grid_fps),
        )
        save_rollout_grid(
            grid_data["failure"],
            path=str(run_dir / "artifacts" / f"failure_grid_{cfg.eval.grid_size}x{cfg.eval.grid_size}.mp4"),
            grid_size=int(cfg.eval.grid_size),
            fps=int(cfg.eval.grid_fps),
        )

    if cfg.eval.record:
        rollout_dir = run_dir / "artifacts" / "rollouts"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        for seed in range(min(5, int(cfg.eval.n_episodes))):
            record_episode(
                model=model,
                processor=processor,
                cfg=cfg,
                make_adapter=adapter_maker,
                path=str(rollout_dir / f"ep_{seed}.mp4"),
                seed=seed,
            )

    print(f"Run artifacts saved under: {run_dir}")
    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "plot_path": str(plot_path),
    }
