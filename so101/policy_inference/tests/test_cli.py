from __future__ import annotations

import sys

from so101.policy_inference import __main__ as cli_module


def test_default_cli_path_is_hardware_free(monkeypatch) -> None:
    calls: list[str] = []
    configs = []
    monkeypatch.setattr(
        cli_module,
        "run_synthetic",
        lambda config, **kwargs: (
            configs.append(config)
            or calls.append("synthetic")
            or {"hardware_opened": False}
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "run_hardware",
        lambda *args, **kwargs: calls.append("hardware"),
    )
    monkeypatch.setattr(
        sys, "argv", ["so101.policy_inference", "--no-rerun", "--duration", "0.1"]
    )

    cli_module.cli()

    assert calls == ["synthetic"]
    assert configs[0].rtc.sampling_noise == "zero"


def test_camera_cli_without_motor_flag_uses_camera_dry_run(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        cli_module,
        "run_camera_dry",
        lambda *args, **kwargs: (
            calls.append("camera_dry") or {"hardware_opened": False}
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "run_hardware",
        lambda *args, **kwargs: calls.append("hardware"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "so101.policy_inference",
            "--camera",
            "wrist=1:180",
            "--camera",
            "base=0:0",
            "--no-rerun",
        ],
    )

    cli_module.cli()

    assert calls == ["camera_dry"]


def test_legacy_random_noise_flag_remains_compatible() -> None:
    args = cli_module.build_parser().parse_args(["--random-noise-each-chunk"])

    assert args.sampling_noise == "random"
