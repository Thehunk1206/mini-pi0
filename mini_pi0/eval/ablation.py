from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from mini_pi0.config.schema import RootConfig
from mini_pi0.eval.runner import run_eval
from mini_pi0.utils.runs import append_jsonl, create_run_dir


def run_eval_ablation(
    cfg: RootConfig,
    execute_steps_values: list[int],
    flow_steps_values: list[int],
    smoothing_values: list[float],
) -> dict[str, Any]:
    """Run a small matrix ablation over eval rollout hyperparameters.

    Args:
        cfg: Base eval config.
        execute_steps_values: Candidate execute-steps values.
        flow_steps_values: Candidate flow integration step counts.
        smoothing_values: Candidate action smoothing alphas.

    Returns:
        Summary payload containing all trial metrics and top-ranked entries.
    """

    root = create_run_dir(cfg.experiment.runs_root, f"{cfg.experiment.name}-ablation")
    rows: list[dict[str, Any]] = []
    trial_idx = 0

    for execute_steps in execute_steps_values:
        for flow_steps in flow_steps_values:
            for alpha in smoothing_values:
                trial_idx += 1
                trial_cfg = copy.deepcopy(cfg)
                trial_cfg.eval.execute_steps = int(execute_steps)
                trial_cfg.eval.n_flow_steps = int(flow_steps)
                trial_cfg.eval.action_smoothing_alpha = float(alpha)
                trial_cfg.eval.run_dir = str(root / f"trial-{trial_idx:03d}")
                print(
                    "[ablation] trial "
                    f"{trial_idx:03d} | execute_steps={execute_steps} n_flow_steps={flow_steps} "
                    f"smoothing={alpha:.3f}",
                    flush=True,
                )
                out = run_eval(trial_cfg)
                summary = dict(out["summary"])
                row = {
                    "trial": trial_idx,
                    "execute_steps": int(execute_steps),
                    "n_flow_steps": int(flow_steps),
                    "action_smoothing_alpha": float(alpha),
                    "run_dir": str(out["run_dir"]),
                    **summary,
                }
                rows.append(row)
                append_jsonl(root / "metrics" / "ablation_metrics.jsonl", row)

    sorted_rows = sorted(
        rows,
        key=lambda x: (
            float(x.get("success_rate", 0.0)),
            float(x.get("reward_mean", 0.0)),
            -float(x.get("episode_len_mean", 0.0)),
        ),
        reverse=True,
    )
    report = {
        "run_dir": str(root),
        "n_trials": len(rows),
        "results": rows,
        "top5": sorted_rows[:5],
    }
    out_json = root / "metrics" / "ablation_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[ablation] Top trials:")
    for row in sorted_rows[:5]:
        print(
            "  "
            f"trial={row['trial']:03d} "
            f"sr={100.0 * float(row.get('success_rate', 0.0)):.1f}% "
            f"reward={float(row.get('reward_mean', 0.0)):.2f} "
            f"len={float(row.get('episode_len_mean', 0.0)):.1f} "
            f"(exec={row['execute_steps']} flow={row['n_flow_steps']} smooth={row['action_smoothing_alpha']:.3f})"
        )
    print(f"[ablation] Artifacts: {root}")
    return report

