from __future__ import annotations

import h5py
import numpy as np
import pytest

from mini_pi0.config.io import load_config
from mini_pi0.dataset.maniskill_collect import collect_maniskill_demos
from mini_pi0.dataset.maniskill_collectors.common import EpisodeBuffer


class _DummyCollector:
    name = "dummy_collector"

    def supports(self, cfg):  # pragma: no cover - not used in these tests
        return True

    def collect_episode(self, req):
        obs = [{k: np.zeros((1,), dtype=np.float32) for k in (req.image_keys + req.state_keys)}]
        ep = EpisodeBuffer(
            obs=obs,
            actions=[np.zeros((7,), dtype=np.float32)],
            rewards=[1.0],
            dones=[1],
            info_rows=[
                {
                    "success_fraction": 1.0,
                    "reward_total": 1.0,
                    "reward_progress": 0.5,
                    "reward_place": 0.2,
                    "reward_terminal": 0.1,
                    "reward_shaping": 0.1,
                    "reward_penalties": 0.0,
                    "reward_step": 0.0,
                }
            ],
        )
        return ep, {"success": True, "success_fraction": 1.0, "placed_count": 1, "total_objects": 1}

    def collect_vectorized(self, req, episodes_target: int):  # pragma: no cover - not used in these tests
        raise NotImplementedError

    def finalize_episode(self, final_info):
        return final_info


def test_collect_append_adds_new_demos(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cfg = load_config("examples/configs/maniskill3_multiobject_tray.yaml")
    monkeypatch.setattr("mini_pi0.dataset.maniskill_collect.resolve_collector", lambda _cfg, _name: _DummyCollector())
    out_path = tmp_path / "tray_append.hdf5"

    first = collect_maniskill_demos(
        cfg,
        out_hdf5=str(out_path),
        num_episodes=1,
        max_steps=5,
        only_success=True,
        collector_backend="scripted",
        collector_name=None,
        num_envs=1,
        overwrite=False,
        append=False,
    )
    second = collect_maniskill_demos(
        cfg,
        out_hdf5=str(out_path),
        num_episodes=1,
        max_steps=5,
        only_success=True,
        collector_backend="scripted",
        collector_name=None,
        num_envs=1,
        overwrite=False,
        append=True,
    )

    assert first["episodes_saved"] == 1
    assert second["episodes_saved"] == 1

    with h5py.File(out_path, "r") as h5:
        demos = sorted(h5["data"].keys())
        assert demos == ["demo_0", "demo_1"]
        assert int(h5["data"].attrs["total"]) == 2


def test_collect_append_conflicts_with_overwrite(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cfg = load_config("examples/configs/maniskill3_multiobject_tray.yaml")
    monkeypatch.setattr("mini_pi0.dataset.maniskill_collect.resolve_collector", lambda _cfg, _name: _DummyCollector())
    out_path = tmp_path / "tray_append_conflict.hdf5"

    with pytest.raises(ValueError, match="append"):
        collect_maniskill_demos(
            cfg,
            out_hdf5=str(out_path),
            num_episodes=1,
            max_steps=5,
            only_success=True,
            collector_backend="scripted",
            collector_name=None,
            num_envs=1,
            overwrite=True,
            append=True,
        )
