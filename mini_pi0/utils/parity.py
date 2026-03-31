from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mini_pi0.config.schema import RootConfig, effective_image_keys, to_dict


@dataclass
class ParityIssue:
    """One checkpoint/runtime parity issue."""

    key: str
    checkpoint: Any
    runtime: Any
    level: str = "error"


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries into dotted-key form."""

    if not isinstance(obj, dict):
        return {prefix: obj}
    out: dict[str, Any] = {}
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def config_diff(before: RootConfig | dict[str, Any], after: RootConfig | dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return dotted-key diff between two configs."""

    b_dict = to_dict(before) if isinstance(before, RootConfig) else before
    a_dict = to_dict(after) if isinstance(after, RootConfig) else after
    b_flat = _flatten(b_dict)
    a_flat = _flatten(a_dict)
    keys = sorted(set(b_flat.keys()) | set(a_flat.keys()))
    out: dict[str, dict[str, Any]] = {}
    for key in keys:
        bv = b_flat.get(key, None)
        av = a_flat.get(key, None)
        if bv != av:
            out[key] = {"before": bv, "after": av}
    return out


def build_checkpoint_parity_report(cfg: RootConfig, ckpt: dict[str, Any]) -> dict[str, Any]:
    """Check runtime config against checkpoint metadata and return report."""

    issues: list[ParityIssue] = []
    warnings: list[str] = []

    ckpt_backend = ckpt.get("sim_backend")
    if ckpt_backend is not None and str(ckpt_backend) != str(cfg.simulator.backend):
        issues.append(
            ParityIssue(
                key="simulator.backend",
                checkpoint=ckpt_backend,
                runtime=cfg.simulator.backend,
            )
        )

    sim_cfg = ckpt.get("sim_config")
    if isinstance(sim_cfg, dict):
        for key in ("task", "robot", "controller"):
            if key not in sim_cfg:
                continue
            ckpt_v = sim_cfg[key]
            run_v = getattr(cfg.simulator, key, None)
            if ckpt_v != run_v:
                issues.append(ParityIssue(key=f"simulator.{key}", checkpoint=ckpt_v, runtime=run_v))

    model_cfg = ckpt.get("model_config")
    if isinstance(model_cfg, dict):
        for key in (
            "action_dim",
            "prop_dim",
            "obs_mode",
            "vision_dim",
            "chunk_size",
            "cond_dim",
            "d_model",
            "nhead",
            "nlayers",
        ):
            if key not in model_cfg:
                continue
            ckpt_v = model_cfg[key]
            run_v = getattr(cfg.model, key, None)
            if ckpt_v != run_v:
                issues.append(ParityIssue(key=f"model.{key}", checkpoint=ckpt_v, runtime=run_v))
    else:
        warnings.append("checkpoint missing model_config metadata")

    robot_cfg = ckpt.get("robot_config")
    if isinstance(robot_cfg, dict):
        for key in ("action_dim",):
            if key not in robot_cfg:
                continue
            ckpt_v = robot_cfg[key]
            run_v = getattr(cfg.robot, key, None)
            if ckpt_v != run_v:
                issues.append(ParityIssue(key=f"robot.{key}", checkpoint=ckpt_v, runtime=run_v))
        if "image_keys" in robot_cfg:
            ckpt_v = list(robot_cfg["image_keys"])
            run_v = effective_image_keys(cfg.robot)
            if ckpt_v != run_v:
                issues.append(ParityIssue(key="robot.image_keys", checkpoint=ckpt_v, runtime=run_v))
        elif "image_key" in robot_cfg:
            ckpt_v = robot_cfg["image_key"]
            run_v = cfg.robot.image_key
            if ckpt_v != run_v:
                issues.append(ParityIssue(key="robot.image_key", checkpoint=ckpt_v, runtime=run_v))

    vision_cfg = ckpt.get("vision_config")
    if str(cfg.model.obs_mode).strip().lower() in {"feature", "precomputed", "features"}:
        if isinstance(vision_cfg, dict):
            for key in ("backend", "model_name", "image_size"):
                if key not in vision_cfg:
                    continue
                ckpt_v = vision_cfg[key]
                run_v = getattr(cfg.vision, key, None)
                if ckpt_v != run_v:
                    issues.append(ParityIssue(key=f"vision.{key}", checkpoint=ckpt_v, runtime=run_v))
        else:
            warnings.append("checkpoint missing vision_config metadata for feature-mode model")

    return {
        "checkpoint_path": None,
        "issues": [issue.__dict__ for issue in issues],
        "warnings": warnings,
        "ok": len(issues) == 0,
    }


def format_parity_issues(issues: list[dict[str, Any]]) -> str:
    """Create readable parity issue summary text."""

    lines = []
    for issue in issues:
        lines.append(
            f"- {issue.get('key')}: checkpoint={issue.get('checkpoint')!r} runtime={issue.get('runtime')!r}"
        )
    return "\n".join(lines)
