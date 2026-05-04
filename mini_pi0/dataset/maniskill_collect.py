from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import h5py

from mini_pi0.config.schema import RootConfig, effective_image_keys, effective_state_keys
from mini_pi0.dataset.maniskill_collectors import _multi_object_tray as _registered_collectors  # noqa: F401
from mini_pi0.dataset.maniskill_collectors.common import summarize_collection_stats, write_episode
from mini_pi0.dataset.maniskill_collectors.interfaces import CollectorRequest
from mini_pi0.dataset.maniskill_collectors.registry import resolve_collector


def _episode_stats_row(num_samples: int, final_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_samples": int(num_samples),
        "success_bool": int(bool(final_info.get("success", False))),
        "final_success_fraction": float(final_info.get("success_fraction", 0.0)),
        "placed_count": int(final_info.get("placed_count", 0)),
        "total_objects": int(final_info.get("total_objects", 0)),
    }


def _next_demo_index(data_group: h5py.Group) -> int:
    """Return next available demo index under an HDF5 `data` group."""
    max_idx = -1
    for key in data_group.keys():
        if not key.startswith("demo_"):
            continue
        suffix = key[5:]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return max_idx + 1


def collect_maniskill_demos(
    cfg: RootConfig,
    *,
    out_hdf5: str,
    num_episodes: int,
    max_steps: int,
    only_success: bool,
    collector_backend: str = "scripted",
    collector_name: str | None = None,
    num_envs: int = 1,
    overwrite: bool = False,
    append: bool = False,
) -> dict[str, Any]:
    """Collect ManiSkill demonstrations into robomimic-style HDF5 via task plugin collectors."""
    out_path = Path(out_hdf5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_keys = effective_image_keys(cfg.robot)
    state_keys = effective_state_keys(cfg.robot)

    num_envs = int(max(1, num_envs))
    collector = resolve_collector(cfg, collector_name)

    if append and overwrite:
        raise ValueError("`append` and `overwrite` cannot both be enabled.")

    if out_path.exists() and (not overwrite) and (not append):
        raise FileExistsError(f"Output HDF5 already exists: {out_path}. Pass --overwrite to replace it.")

    saved = 0
    trials = 0
    collected_rows: list[dict[str, Any]] = []
    file_mode = "a" if append else "w"
    t0 = time.perf_counter()

    with h5py.File(out_path, file_mode) as h5:
        data_group = h5.require_group("data")
        start_demo_idx = _next_demo_index(data_group) if append else 0
        existing_total = int(data_group.attrs.get("total", 0)) if append else 0
        data_group.attrs["total"] = existing_total
        data_group.attrs["env_args"] = json.dumps(
            {
                "env_name": cfg.simulator.task,
                "env_type": "maniskill3_custom",
                "env_kwargs": dict(cfg.simulator.env_kwargs),
                "collector_name": collector.name,
            }
        )

        while saved < int(num_episodes):
            trials += 1
            if trials > max(10, int(num_episodes) * 5):
                break
            trial_t0 = time.perf_counter()
            print(
                "[collect] trial_start "
                f"trial={trials} saved={saved}/{int(num_episodes)} "
                f"num_envs={int(num_envs)} backend={collector_backend} only_success={bool(only_success)}",
                flush=True,
            )

            ep_cfg = copy.deepcopy(cfg)
            ep_cfg.experiment.seed = int(cfg.experiment.seed + trials - 1)
            req = CollectorRequest(
                cfg=ep_cfg,
                image_keys=image_keys,
                state_keys=state_keys,
                num_envs=int(num_envs),
                max_steps=int(max_steps),
                only_success=bool(only_success),
                backend=str(collector_backend),
            )

            if num_envs > 1:
                episodes = collector.collect_vectorized(req, episodes_target=int(num_episodes) - int(saved))
                saved_before = int(saved)
                for ep, finfo in episodes:
                    final_info = collector.finalize_episode(dict(finfo))
                    final_info["collector_type"] = str(collector_backend if collector_backend else "scripted")
                    if only_success and not bool(final_info.get("success", False)):
                        continue
                    write_episode(data_group, start_demo_idx + saved, ep, final_info)
                    data_group.attrs["total"] = int(data_group.attrs.get("total", 0)) + len(ep.actions)
                    collected_rows.append(_episode_stats_row(len(ep.actions), final_info))
                    saved += 1
                print(
                    "[collect] trial_end "
                    f"trial={trials} added={saved - saved_before} "
                    f"saved={saved}/{int(num_episodes)} elapsed_s={time.perf_counter() - trial_t0:.1f}",
                    flush=True,
                )
                continue

            ep, finfo = collector.collect_episode(req)
            final_info = collector.finalize_episode(dict(finfo))
            final_info["collector_type"] = str(collector_backend if collector_backend else "scripted")

            if only_success and not bool(final_info.get("success", False)):
                continue

            write_episode(data_group, start_demo_idx + saved, ep, final_info)
            data_group.attrs["total"] = int(data_group.attrs.get("total", 0)) + len(ep.actions)
            collected_rows.append(_episode_stats_row(len(ep.actions), final_info))
            saved += 1
            print(
                "[collect] trial_end "
                f"trial={trials} added=1 saved={saved}/{int(num_episodes)} "
                f"success={bool(final_info.get('success', False))} "
                f"success_fraction={float(final_info.get('success_fraction', 0.0)):.3f} "
                f"elapsed_s={time.perf_counter() - trial_t0:.1f}",
                flush=True,
            )

    out = {
        "path": str(out_path),
        "episodes_saved": int(saved),
        "trials": int(trials),
        "only_success": bool(only_success),
        "collector_backend": str(collector_backend),
        "collector_name": collector.name,
        "num_envs": int(num_envs),
        "append": bool(append),
        "stats": summarize_collection_stats(collected_rows),
    }
    print(
        "[collect] done "
        f"saved={out['episodes_saved']}/{int(num_episodes)} trials={out['trials']} "
        f"elapsed_s={time.perf_counter() - t0:.1f}",
        flush=True,
    )
    return out
