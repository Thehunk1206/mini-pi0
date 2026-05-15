from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

from mini_pi0.config.io import dump_config
from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys
from mini_pi0.dataset.obs_processor import ObsProcessor
from mini_pi0.eval.core import _maybe_binarize_gripper_action
from mini_pi0.models.registry import load_checkpoint, make_model
from mini_pi0.sim.registry import make_sim_adapter
from mini_pi0.utils.device import resolve_device
from mini_pi0.utils.parity import build_checkpoint_parity_report, config_diff, format_parity_issues
from mini_pi0.utils.runs import create_run_dir


def _reshape_action(action: np.ndarray, target_dim: int) -> np.ndarray:
    """Resize action vector to match adapter action space dimension.

    Args:
        action: Input action vector.
        target_dim: Required output dimension.

    Returns:
        Vector with length ``target_dim``.
    """

    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] == target_dim:
        return a
    if a.shape[0] > target_dim:
        return a[:target_dim]
    out = np.zeros((target_dim,), dtype=np.float32)
    out[: a.shape[0]] = a
    return out


def _blend_with_previous_tail(
    chunk: np.ndarray,
    previous_tail: np.ndarray | None,
    blend: float,
) -> np.ndarray:
    """Blend a new chunk prefix with the unused previous chunk tail."""

    alpha = float(max(0.0, min(1.0, blend)))
    if previous_tail is None or alpha <= 0.0:
        return chunk
    out = np.asarray(chunk, dtype=np.float32).copy()
    tail = np.asarray(previous_tail, dtype=np.float32)
    n = min(out.shape[0], tail.shape[0])
    if n > 0:
        out[:n] = (1.0 - alpha) * out[:n] + alpha * tail[:n]
    return out


def _resolve_deploy_rollout_controls(cfg: RootConfig, env_steps_done: int) -> tuple[int, int, float]:
    """Resolve execute/flow/smoothing controls with optional warmup overrides."""

    execute_steps = int(max(1, cfg.deploy.execute_steps))
    n_flow_steps = int(max(1, cfg.deploy.n_flow_steps))
    smooth_alpha = float(max(0.0, min(1.0, getattr(cfg.deploy, "action_smoothing_alpha", 0.0))))

    warmup_steps = int(max(0, getattr(cfg.deploy, "stability_warmup_steps", 0)))
    if warmup_steps <= 0 or env_steps_done >= warmup_steps:
        return execute_steps, n_flow_steps, smooth_alpha

    warm_exec = getattr(cfg.deploy, "stability_warmup_execute_steps", None)
    warm_flow = getattr(cfg.deploy, "stability_warmup_n_flow_steps", None)
    warm_smooth = getattr(cfg.deploy, "stability_warmup_action_smoothing_alpha", None)
    if warm_exec is not None:
        execute_steps = int(max(1, warm_exec))
    if warm_flow is not None:
        n_flow_steps = int(max(1, warm_flow))
    if warm_smooth is not None:
        smooth_alpha = float(max(0.0, min(1.0, warm_smooth)))
    return execute_steps, n_flow_steps, smooth_alpha


def _inject_model_cfg_from_checkpoint(cfg: RootConfig, ckpt: dict[str, Any]) -> None:
    """Merge checkpoint model metadata into runtime config.

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


def run_deploy_sim(cfg: RootConfig) -> dict[str, Any]:
    """Run closed-loop simulation deployment using a trained checkpoint.

    Args:
        cfg: Resolved root configuration.

    Returns:
        Summary dictionary with success, reward, and artifact paths.

    Raises:
        NotImplementedError: If non-simulation deployment mode is requested.
        ValueError: If checkpoint format is invalid.
    """

    if str(cfg.deploy.mode).lower() != "sim":
        raise NotImplementedError(
            "Only deploy.mode=sim is implemented in this package."
        )

    run_dir = create_run_dir(cfg.experiment.runs_root, f"{cfg.experiment.name}-deploy")
    requested_cfg = copy.deepcopy(cfg)

    device = resolve_device(cfg.deploy.device)
    ckpt = load_checkpoint(cfg.deploy.checkpoint, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint format is invalid.")

    parity = build_checkpoint_parity_report(requested_cfg, ckpt)
    parity["checkpoint_path"] = str(cfg.deploy.checkpoint)
    strict = bool(getattr(cfg.deploy, "strict_parity", True))
    if parity["issues"]:
        msg = "Checkpoint/runtime parity mismatches detected:\n" + format_parity_issues(parity["issues"])
        if strict:
            raise ValueError(msg + "\nDisable with --set deploy.strict_parity=false if intentional.")
        print(f"[deploy] WARNING: {msg}", flush=True)
    for w in parity.get("warnings", []):
        print(f"[deploy] WARNING: {w}", flush=True)

    _inject_model_cfg_from_checkpoint(cfg, ckpt)
    runtime_cfg = copy.deepcopy(cfg)
    print(
        "[deploy] Preflight | "
        f"backend={cfg.simulator.backend} task={cfg.simulator.task} robot={cfg.simulator.robot} "
        f"controller={cfg.simulator.controller} obs_mode=image "
        f"image_keys={effective_image_keys(cfg.robot)} strict_parity={strict}",
        flush=True,
    )
    dump_config(run_dir / "metrics" / "deploy_config_requested.yaml", requested_cfg)
    dump_config(run_dir / "metrics" / "deploy_config_runtime.yaml", runtime_cfg)
    with (run_dir / "metrics" / "deploy_provenance.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(cfg.deploy.checkpoint),
                "strict_parity": strict,
                "parity": parity,
                "requested_to_runtime_diff": config_diff(requested_cfg, runtime_cfg),
            },
            f,
            indent=2,
        )

    model = make_model(cfg).to(device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    image_keys = effective_image_keys(cfg.robot)
    stats_path = cfg.deploy.action_stats_path or cfg.data.action_stats_path
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

    sim_cfg = copy.deepcopy(cfg)
    sim_cfg.experiment.seed = int(cfg.experiment.seed)
    adapter = make_sim_adapter(sim_cfg)

    obs = adapter.reset(seed=int(cfg.experiment.seed))
    processor.reset_history(obs)
    action_buffer: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    reward_sum = 0.0
    success = False

    lo, hi = adapter.action_spec()
    action_dim = int(np.asarray(lo).reshape(-1).shape[0])
    action_scale = np.asarray(cfg.deploy.action_scale, dtype=np.float32).reshape(-1) if cfg.deploy.action_scale else None
    prev_action: np.ndarray | None = None
    previous_chunk_tail: np.ndarray | None = None

    for step_idx in range(int(cfg.deploy.max_steps)):
        if not action_buffer:
            execute_steps, n_flow_steps, smooth_alpha = _resolve_deploy_rollout_controls(cfg, step_idx)
            img, prop = processor.obs_to_tensors(obs)
            with torch.no_grad():
                chunk = model.sample(
                    img,
                    prop,
                    n_steps=n_flow_steps,
                    solver=str(getattr(cfg.deploy, "flow_solver", "euler")),
                ).squeeze(0)
            chunk = processor.denormalize(chunk).detach().cpu().numpy()
            chunk = _blend_with_previous_tail(
                chunk,
                previous_chunk_tail,
                float(getattr(cfg.deploy, "chunk_overlap_blend", 0.0)),
            )
            previous_chunk_tail = chunk[execute_steps:].copy()
            proposed = []
            for a in chunk[:execute_steps]:
                x = _reshape_action(a, target_dim=action_dim)
                if action_scale is not None and action_scale.shape[0] == action_dim:
                    x = x * action_scale
                if prev_action is not None and smooth_alpha > 0.0:
                    x = (1.0 - smooth_alpha) * x + smooth_alpha * prev_action
                x = _maybe_binarize_gripper_action(x, cfg.deploy)
                x = np.clip(x, lo, hi).astype(np.float32)
                proposed.append(x)
            action_buffer = proposed

        action = action_buffer.pop(0)
        prev_action = action.copy()
        step = adapter.step(action)
        obs = step.obs
        reward_sum += float(step.reward)
        success = adapter.check_success(info=step.info, obs=step.obs)

        if cfg.deploy.record_path:
            frame = adapter.render(camera=cfg.simulator.camera_names[0], width=512, height=512)
            frame = np.asarray(frame)
            if frame.ndim == 3:
                if frame.dtype != np.uint8:
                    if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
                        frame = frame * 255.0
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                frames.append(frame)

        if step.done or success:
            break

    adapter.close()

    if cfg.deploy.record_path and frames:
        rec_path = Path(cfg.deploy.record_path)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(rec_path, fps=20) as writer:
            for frame in frames:
                writer.append_data(frame)
    else:
        rec_path = None

    summary = {
        "run_dir": str(run_dir),
        "success": bool(success),
        "total_reward": float(reward_sum),
        "recording": str(rec_path) if rec_path is not None else None,
    }
    with (run_dir / "metrics" / "deploy_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Deploy(sim) success={summary['success']} reward={summary['total_reward']:.3f}")
    print(f"Run artifacts saved under: {run_dir}")
    return summary
