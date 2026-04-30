from __future__ import annotations

import uuid

from mini_pi0.config.io import load_config
import mini_pi0.dataset.maniskill_collectors  # noqa: F401
from mini_pi0.dataset.maniskill_collectors.interfaces import CollectorRequest
from mini_pi0.dataset.maniskill_collectors.registry import get_collector, register_collector, resolve_collector


class _DummyCollector:
    def __init__(self, name: str, task_match: str):
        self.name = name
        self._task = task_match

    def supports(self, cfg):
        return str(cfg.simulator.task) == self._task

    def collect_episode(self, req: CollectorRequest):  # pragma: no cover - not used in registry tests
        raise NotImplementedError

    def collect_vectorized(self, req: CollectorRequest, episodes_target: int):  # pragma: no cover
        raise NotImplementedError

    def finalize_episode(self, final_info):  # pragma: no cover
        return final_info


def test_registry_explicit_name_lookup():
    cfg = load_config("examples/configs/maniskill3_multiobject_tray.yaml")
    name = f"dummy_{uuid.uuid4().hex[:8]}"
    collector = register_collector(_DummyCollector(name=name, task_match="never-match"))
    got = resolve_collector(cfg, collector_name=name)
    assert got is collector
    assert get_collector(name) is collector


def test_registry_infer_from_task():
    cfg = load_config("examples/configs/maniskill3_multiobject_tray.yaml")
    got = resolve_collector(cfg, collector_name=None)
    assert got.name == "mini_pi0_multiobject_tray"


def test_registry_missing_collector_errors():
    cfg = load_config("examples/configs/maniskill3_multiobject_tray.yaml")
    try:
        resolve_collector(cfg, collector_name="does_not_exist")
    except ValueError as exc:
        assert "Unknown collector_name" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown collector")


def test_builtin_collector_contract():
    collector = get_collector("mini_pi0_multiobject_tray")
    for attr in ("name", "supports", "collect_episode", "collect_vectorized", "finalize_episode"):
        assert hasattr(collector, attr), f"missing collector attribute: {attr}"
