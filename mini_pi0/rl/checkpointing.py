"""Atomic ReinFlow checkpoints with optimizer, reference, and RNG state."""

from __future__ import annotations

import importlib.metadata
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mini_pi0.config.schema import RootConfig, to_dict
from mini_pi0.rl.exceptions import ReinFlowCheckpointError
from mini_pi0.rl.flow_policy import ReinFlowActorCritic
from mini_pi0.rl.flow_ppo import ReinFlowPPOUpdater


@dataclass(frozen=True)
class ResumeState:
    """Counters restored before the next PPO update."""

    next_update: int
    primitive_steps: int
    best_eval_success: float
    latest_metrics: dict[str, object]


def save_reinflow_checkpoint(
    path: Path,
    *,
    cfg: RootConfig,
    policy: ReinFlowActorCritic,
    updater: ReinFlowPPOUpdater,
    next_update: int,
    primitive_steps: int,
    best_eval_success: float,
    metrics: dict[str, object],
    environment_manifest: dict[str, object],
) -> None:
    """Atomically save all state required for ReinFlow continuation."""

    payload = {
        "format_version": 1,
        "rl_algorithm": "reinflow_ppo",
        "model": policy.policy.state_dict(),
        "actor_critic": policy.state_dict(),
        "reference": updater.reference.state_dict() if updater.reference is not None else None,
        "actor_optimizer": updater.actor_optimizer.state_dict(),
        "critic_optimizer": updater.critic_optimizer.state_dict(),
        "actor_scheduler": updater.actor_scheduler.state_dict(),
        "critic_scheduler": updater.critic_scheduler.state_dict(),
        "next_update": int(next_update),
        "primitive_steps": int(primitive_steps),
        "best_eval_success": float(best_eval_success),
        "resolved_config": to_dict(cfg),
        "action_stats": _read_action_stats(cfg.rl.action_stats_path),
        "rng_state": _capture_rng_state(next(policy.parameters()).device),
        "git_commit": _git_commit(),
        "versions": _dependency_versions(),
        "environment_manifest": environment_manifest,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_reinflow_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate a ReinFlow checkpoint payload."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("rl_algorithm") != "reinflow_ppo":
        raise ReinFlowCheckpointError(f"Not a ReinFlow checkpoint: {path}")
    required = {
        "actor_critic",
        "actor_optimizer",
        "critic_optimizer",
        "actor_scheduler",
        "critic_scheduler",
        "rng_state",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ReinFlowCheckpointError(f"ReinFlow checkpoint is missing: {', '.join(missing)}")
    return payload


def restore_reinflow_checkpoint(
    payload: dict[str, Any],
    *,
    policy: ReinFlowActorCritic,
    updater: ReinFlowPPOUpdater,
) -> ResumeState:
    """Restore model, optimizers, schedulers, reference, and RNG state."""

    try:
        policy.load_state_dict(payload["actor_critic"], strict=True)
        updater.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        updater.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        updater.actor_scheduler.load_state_dict(payload["actor_scheduler"])
        updater.critic_scheduler.load_state_dict(payload["critic_scheduler"])
        reference_state = payload.get("reference")
        if reference_state is not None:
            if updater.reference is None:
                raise ReinFlowCheckpointError("Checkpoint contains a reference policy but config disables it.")
            updater.reference.load_state_dict(reference_state, strict=True)
        elif updater.reference is not None:
            raise ReinFlowCheckpointError("Config enables a reference policy but checkpoint does not contain one.")
        _restore_rng_state(payload["rng_state"], next(policy.parameters()).device)
    except ReinFlowCheckpointError:
        raise
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ReinFlowCheckpointError(f"Failed to restore ReinFlow state: {exc}") from exc
    metrics = payload.get("metrics")
    return ResumeState(
        next_update=int(payload.get("next_update", 0)),
        primitive_steps=int(payload.get("primitive_steps", 0)),
        best_eval_success=float(payload.get("best_eval_success", float("-inf"))),
        latest_metrics=dict(metrics) if isinstance(metrics, dict) else {},
    )


def materialize_embedded_action_stats(payload: dict[str, Any], destination: Path) -> Path | None:
    """Write embedded normalization stats for the observation processor."""

    stats = payload.get("action_stats")
    if stats is None:
        return None
    if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
        raise ReinFlowCheckpointError("Embedded action statistics are malformed.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return destination


def _capture_rng_state(device: torch.device) -> dict[str, object]:
    """Capture Python, NumPy, and Torch random generators."""

    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng_state(state: dict[str, object], device: torch.device) -> None:
    """Restore all random generators present in a checkpoint."""

    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and device.type == "cuda":
        torch.cuda.set_rng_state(cuda_state, device)  # type: ignore[arg-type]


def _read_action_stats(path: str | None) -> dict[str, object] | None:
    """Read optional dataset normalization stats into the checkpoint."""

    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReinFlowCheckpointError(f"Action statistics must be a JSON object: {path}")
    return payload


def _git_commit() -> str | None:
    """Return the current Git commit when available."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _dependency_versions() -> dict[str, str]:
    """Record compact dependency versions useful for reproducing a run."""

    versions = {"python_torch": str(torch.__version__), "numpy": str(np.__version__)}
    for package in ("mini-pi0", "gymnasium", "mani-skill", "isaaclab"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions
