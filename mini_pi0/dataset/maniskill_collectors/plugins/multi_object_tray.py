"""Collector plugin for the MiniPi0 multi-object bowl/tray ManiSkill task.

This plugin binds task-specific collection behavior to the generic collection
orchestrator:
- task matching (`supports`)
- backend routing (`scripted` vs `mplib`)
- vectorized collection policy
- final per-episode metric normalization
"""

from __future__ import annotations

from typing import Any

from mini_pi0.config.schema import RootConfig
from mini_pi0.dataset.maniskill_collectors.backends import (
    collect_single_mplib_episode,
    collect_single_scripted_episode,
    collect_vectorized_scripted_episodes,
    mplib_runtime_check,
)
from mini_pi0.dataset.maniskill_collectors.interfaces import CollectorRequest
from mini_pi0.dataset.maniskill_collectors.registry import register_collector


class MiniPi0MultiObjectTrayCollector:
    """Task collector for `MiniPi0MultiObjectTray-v1` and legacy aliases."""

    name = "mini_pi0_multiobject_tray"

    def supports(self, cfg: RootConfig) -> bool:
        """Return True when this collector should handle the provided config.

        Args:
            cfg: Runtime config carrying `simulator.task`.

        Returns:
            True when task id matches this collector's supported aliases.
        """
        task = str(cfg.simulator.task or "").strip().lower()
        return task in {
            "minipi0multiobjecttray-v1",
            "mini_pi0_multiobject",
            "custom",
        }

    def collect_episode(self, req: CollectorRequest):
        """Collect a single episode using backend preference in `req.backend`.

        Args:
            req: Normalized collector request from orchestrator.

        Returns:
            `(EpisodeBuffer, final_info)` for one episode.

        Behavior:
            - if backend is `mplib` and runtime check passes, use
              motion-planning backend
            - otherwise use scripted oracle backend
        """
        backend = str(req.backend or "scripted").lower()
        if backend == "mplib" and mplib_runtime_check():
            return collect_single_mplib_episode(
                req.cfg,
                max_steps=int(req.max_steps),
                image_keys=req.image_keys,
                state_keys=req.state_keys,
            )
        return collect_single_scripted_episode(
            req.cfg,
            image_keys=req.image_keys,
            state_keys=req.state_keys,
            max_steps=int(req.max_steps),
        )

    def collect_vectorized(self, req: CollectorRequest, episodes_target: int):
        """Collect vectorized episodes with scripted-first stability policy.

        Args:
            req: Normalized collector request from orchestrator.
            episodes_target: Target number of finalized episodes.

        Returns:
            List of `(EpisodeBuffer, final_info)` episodes.

        Current policy deliberately routes vectorized collection through the
        scripted backend; mplib vectorized execution is not used here.
        """
        return collect_vectorized_scripted_episodes(
            req.cfg,
            image_keys=req.image_keys,
            state_keys=req.state_keys,
            num_envs=int(req.num_envs),
            episodes_target=int(episodes_target),
            max_steps=int(req.max_steps),
            only_success=bool(req.only_success),
        )

    def finalize_episode(self, final_info: dict[str, Any]) -> dict[str, Any]:
        """Normalize final metrics required by shared HDF5 writer and stats.

        Args:
            final_info: Raw final info produced by selected backend.

        Returns:
            Normalized dict with required fields and default fallbacks.
        """
        out = dict(final_info)
        out.setdefault("success", False)
        out.setdefault("success_fraction", 0.0)
        out.setdefault("placed_count", 0)
        out.setdefault("total_objects", 0)
        return out


register_collector(MiniPi0MultiObjectTrayCollector())
