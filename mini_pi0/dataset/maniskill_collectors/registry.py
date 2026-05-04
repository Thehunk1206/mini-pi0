"""Registry and resolution helpers for ManiSkill collector plugins."""

from __future__ import annotations

from typing import Dict

from mini_pi0.config.schema import RootConfig
from .interfaces import TaskCollector

_COLLECTORS: Dict[str, TaskCollector] = {}


def register_collector(collector: TaskCollector) -> TaskCollector:
    """Register a collector by its unique `collector.name`.

    Args:
        collector: Collector plugin instance.

    Returns:
        The same collector instance (convenient for inline registration).

    Raises:
        ValueError: If collector name is missing or already registered.
    """
    name = getattr(collector, "name", "").strip()
    if not name:
        raise ValueError("Collector must define a non-empty `name`.")
    if name in _COLLECTORS:
        raise ValueError(f"Collector already registered: {name}")
    _COLLECTORS[name] = collector
    return collector


def get_collector(name: str) -> TaskCollector:
    """Resolve a collector by explicit name.

    Args:
        name: Collector id passed from CLI/API.

    Returns:
        Registered collector instance.

    Raises:
        ValueError: If `name` is not found in registry.
    """
    key = str(name).strip()
    if key not in _COLLECTORS:
        known = ", ".join(sorted(_COLLECTORS.keys()))
        raise ValueError(f"Unknown collector_name `{key}`. Registered collectors: {known}")
    return _COLLECTORS[key]


def resolve_collector(cfg: RootConfig, collector_name: str | None) -> TaskCollector:
    """Resolve collector with precedence: explicit name, then `supports(cfg)`.

    Args:
        cfg: Runtime config used for inference.
        collector_name: Optional explicit collector id.

    Returns:
        Selected collector instance.

    Raises:
        ValueError: If no collector matches, or multiple collectors match.
    """
    if collector_name is not None and str(collector_name).strip() != "":
        return get_collector(str(collector_name))

    matches = [collector for collector in _COLLECTORS.values() if bool(collector.supports(cfg))]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        known = ", ".join(sorted(_COLLECTORS.keys()))
        raise ValueError(
            f"No collector supports simulator.task={cfg.simulator.task!r}. "
            f"Provide --collector_name explicitly. Registered collectors: {known}"
        )
    names = ", ".join(sorted(c.name for c in matches))
    raise ValueError(
        f"Multiple collectors support simulator.task={cfg.simulator.task!r}: {names}. "
        "Provide --collector_name explicitly."
    )


def list_collectors() -> list[str]:
    """Return sorted list of registered collector names."""
    return sorted(_COLLECTORS.keys())
