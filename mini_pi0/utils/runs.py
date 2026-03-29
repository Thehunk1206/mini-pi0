from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def slugify(text: str) -> str:
    """Convert arbitrary experiment names into filesystem-safe slugs.

    Args:
        text: Raw experiment name.

    Returns:
        Lowercased slug safe for directory names.
    """

    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in text.strip().lower())
    safe = "-".join(filter(None, safe.split("-")))
    return safe or "run"


def create_run_dir(root: str = "runs", experiment_name: str = "default") -> Path:
    """Create sequential run directory with standard subfolders.

    Args:
        root: Root runs directory.
        experiment_name: Experiment slug source string.

    Returns:
        Path to newly created run directory.
    """

    exp_root = Path(root) / slugify(experiment_name)
    exp_root.mkdir(parents=True, exist_ok=True)

    pat = re.compile(r"^run(\d+)$")
    max_idx = 0
    for p in exp_root.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    run_name = f"run{max_idx + 1}"
    run_dir = exp_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_yaml(path: os.PathLike[str] | str, data: Any) -> None:
    """Write dictionary-like data to YAML file.

    Args:
        path: Destination path.
        data: Data payload to serialize.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def append_jsonl(path: os.PathLike[str] | str, item: dict[str, Any]) -> None:
    """Append one JSON object line to a JSONL metrics file.

    Args:
        path: JSONL destination path.
        item: Dictionary row to append.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=True) + "\n")
