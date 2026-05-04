"""Typed interfaces for task-specific ManiSkill collector plugins.

This module defines:
- `CollectorRequest`: normalized runtime inputs passed from orchestrator
- `TaskCollector`: plugin protocol implemented per task/env
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from mini_pi0.config.schema import RootConfig
from .common import EpisodeBuffer


@dataclass
class CollectorRequest:
    """Arguments passed to collector plugins for one collection request.

    Attributes:
        cfg: Resolved runtime config for the current trial/seed.
        image_keys: Canonical image observation keys to save.
        state_keys: Canonical state observation keys to save.
        num_envs: Number of environments requested by caller.
        max_steps: Per-episode step budget.
        only_success: Whether caller intends to keep success-only demos.
        backend: Backend hint from CLI (for example `scripted` or `mplib`).
    """

    cfg: RootConfig
    image_keys: list[str]
    state_keys: list[str]
    num_envs: int
    max_steps: int
    only_success: bool
    backend: str


class TaskCollector(Protocol):
    """Task-specific collection plugin interface."""

    name: str

    def supports(self, cfg: RootConfig) -> bool:
        """Return True when this plugin can handle the provided config.

        Args:
            cfg: Fully resolved runtime config.

        Returns:
            True if this collector should be selected for `cfg`.
        """
        ...

    def collect_episode(self, req: CollectorRequest) -> tuple[EpisodeBuffer, dict[str, Any]]:
        """Collect one single-environment episode.

        Args:
            req: Normalized collection request.

        Returns:
            A tuple of `(episode_buffer, final_info)` where `final_info`
            contains task-level completion metrics.
        """
        ...

    def collect_vectorized(self, req: CollectorRequest, episodes_target: int) -> list[tuple[EpisodeBuffer, dict[str, Any]]]:
        """Collect multiple episodes from vectorized rollouts.

        Args:
            req: Normalized collection request.
            episodes_target: Target number of finalized episodes to return.

        Returns:
            List of `(episode_buffer, final_info)` entries.
        """
        ...

    def finalize_episode(self, final_info: dict[str, Any]) -> dict[str, Any]:
        """Normalize final metrics before writing to HDF5/stat summary.

        Args:
            final_info: Raw per-episode final info produced by backend/env.

        Returns:
            Normalized dict with required keys consumed by writer/stats code.
        """
        ...
