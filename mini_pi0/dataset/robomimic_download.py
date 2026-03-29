from __future__ import annotations

import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any


_TASKS_BY_TYPE: dict[str, set[str]] = {
    "ph": {"lift", "can", "square", "transport", "tool_hang"},
    "mh": {"lift", "can", "square", "transport"},
    "mg": {"lift", "can"},
}

_HDF5_TYPES_BY_DATASET: dict[str, set[str]] = {
    "ph": {"low_dim"},
    "mh": {"low_dim"},
    "mg": {"low_dim_sparse", "low_dim_dense"},
}


def _validate_combo(task: str, dataset_type: str, hdf5_type: str) -> None:
    if dataset_type not in _TASKS_BY_TYPE:
        raise ValueError(f"Unsupported dataset_type '{dataset_type}'.")
    if task not in _TASKS_BY_TYPE[dataset_type]:
        supported = sorted(_TASKS_BY_TYPE[dataset_type])
        raise ValueError(
            f"task='{task}' is not available for dataset_type='{dataset_type}'. "
            f"Supported tasks: {supported}"
        )
    if hdf5_type not in _HDF5_TYPES_BY_DATASET[dataset_type]:
        supported = sorted(_HDF5_TYPES_BY_DATASET[dataset_type])
        raise ValueError(
            f"hdf5_type='{hdf5_type}' is not available for dataset_type='{dataset_type}'. "
            f"Supported hdf5 types: {supported}"
        )


def _build_url(task: str, dataset_type: str, hdf5_type: str, version: str = "v1.5") -> str:
    filename = f"{hdf5_type}_v15.hdf5"
    return (
        "https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/"
        f"{version}/{task}/{dataset_type}/{filename}"
    )


def _download_file(url: str, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def download_robomimic_dataset(
    *,
    task: str,
    dataset_type: str,
    hdf5_type: str,
    download_dir: str,
    version: str = "v1.5",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download one robomimic dataset HDF5 from official Hugging Face hosting.

    Args:
        task: Task key (for example ``lift`` or ``can``).
        dataset_type: Dataset family key (``ph``, ``mh``, ``mg``).
        hdf5_type: HDF5 variant (for example ``low_dim``).
        download_dir: Local root directory where files are saved.
        version: Dataset release branch segment (default ``v1.5``).
        overwrite: If ``True``, overwrite existing file.

    Returns:
        Dictionary with download metadata and local output path.
    """

    task = str(task).strip().lower()
    dataset_type = str(dataset_type).strip().lower()
    hdf5_type = str(hdf5_type).strip().lower()
    version = str(version).strip()
    if not version:
        raise ValueError("version must be a non-empty string (for example 'v1.5').")

    _validate_combo(task=task, dataset_type=dataset_type, hdf5_type=hdf5_type)

    url = _build_url(task=task, dataset_type=dataset_type, hdf5_type=hdf5_type, version=version)
    dest = Path(download_dir) / task / dataset_type / f"{hdf5_type}_v15.hdf5"

    pre_exists = dest.exists()
    _download_file(url=url, destination=dest, overwrite=overwrite)

    output = {
        "task": task,
        "dataset_type": dataset_type,
        "hdf5_type": hdf5_type,
        "version": version,
        "url": url,
        "path": str(dest),
        "downloaded": bool(overwrite or not pre_exists),
        "size_bytes": int(os.path.getsize(dest)),
    }
    sidecar = dest.with_suffix(".download.json")
    sidecar.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output

