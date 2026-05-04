"""Template plugin for adding a new task-specific ManiSkill collector.

Copy this file, rename symbols, implement backend calls, and register the
collector. Keep output schema compatibility by always returning normalized
final metrics in `finalize_episode`.
"""

from __future__ import annotations

from typing import Any

from mini_pi0.config.schema import RootConfig
from mini_pi0.dataset.maniskill_collectors.interfaces import CollectorRequest


class NewTaskCollectorTemplate:
    """Template collector plugin for adding a new ManiSkill task/env.

    Usage:
    1) Copy this file to `plugins/<your_task>.py`.
    2) Rename class + `name`.
    3) Implement single/vectorized collection methods.
    4) Register the collector in your module with `register_collector(...)`.
    5) Import that plugin in `maniskill_collectors/__init__.py`.
    """

    # Stable id used by CLI `--collector_name`.
    name = "replace_with_collector_name"

    # Set the task id(s) your collector handles.
    supported_tasks = {"ReplaceTask-v1"}

    def supports(self, cfg: RootConfig) -> bool:
        """Return True if this collector should handle `cfg.simulator.task`.

        Args:
            cfg: Runtime config carrying simulator/task selection.

        Returns:
            True when current task is listed in `supported_tasks`.
        """
        task = str(cfg.simulator.task or "").strip()
        return task in self.supported_tasks

    def collect_episode(self, req: CollectorRequest):
        """Collect and return one episode buffer plus final episode metrics.

        Args:
            req: Normalized collector request from orchestrator.

        Returns:
            Tuple `(EpisodeBuffer, final_info_dict)`.
        """
        # TODO: Build one episode and return:
        # (EpisodeBuffer, final_info_dict)
        # final_info_dict should include success/progress metrics.
        raise NotImplementedError("Implement collect_episode for your task.")

    def collect_vectorized(self, req: CollectorRequest, episodes_target: int):
        """Collect and return multiple episodes from vectorized env rollouts.

        Args:
            req: Normalized collector request from orchestrator.
            episodes_target: Number of finalized episodes to return.

        Returns:
            List of `(EpisodeBuffer, final_info_dict)` pairs.
        """
        # TODO: Build batched collection and return:
        # list[(EpisodeBuffer, final_info_dict)]
        raise NotImplementedError("Implement collect_vectorized for your task.")

    def finalize_episode(self, final_info: dict[str, Any]) -> dict[str, Any]:
        """Guarantee required fields used by shared writer/stats pipeline.

        Args:
            final_info: Raw final info from backend.

        Returns:
            Normalized final info with required default keys populated.
        """
        # Always guarantee required fields used by shared writer/stats.
        out = dict(final_info)
        out.setdefault("success", False)
        out.setdefault("success_fraction", 0.0)
        out.setdefault("placed_count", 0)
        out.setdefault("total_objects", 0)
        return out
