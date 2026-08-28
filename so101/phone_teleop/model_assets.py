"""Fetch, validate, and serve the authoritative SO-101 visual model."""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


BUCKET_URI = "hf://buckets/lerobot/robot-urdfs/so101"
MODEL_FILENAME = "so101_new_calib.urdf"
EXPECTED_STL_COUNT = 13
COMPLETION_MARKER = ".mini_pi0_complete.json"
DEFAULT_MODEL_CACHE = (
    Path.home() / ".cache" / "huggingface" / "lerobot" / "robot-urdfs" / "so101"
)


def referenced_meshes(urdf_path: Path) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    return sorted(
        {
            mesh.attrib["filename"]
            for mesh in root.findall(".//visual/geometry/mesh")
            if mesh.attrib.get("filename", "").lower().endswith(".stl")
        }
    )


def validate_model_cache(cache_dir: Path = DEFAULT_MODEL_CACHE) -> dict[str, Any]:
    cache = Path(cache_dir)
    urdf_path = cache / MODEL_FILENAME
    if not urdf_path.is_file() or urdf_path.stat().st_size == 0:
        raise FileNotFoundError(f"Official SO-101 URDF is missing: {urdf_path}")
    meshes = referenced_meshes(urdf_path)
    if len(meshes) != EXPECTED_STL_COUNT:
        raise ValueError(
            f"Official SO-101 URDF references {len(meshes)} unique STL files; "
            f"expected {EXPECTED_STL_COUNT}"
        )
    missing = [relative for relative in meshes if not (cache / relative).is_file() or (cache / relative).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Official SO-101 model is incomplete: {missing}")
    return {
        "cache_dir": str(cache),
        "urdf": MODEL_FILENAME,
        "meshes": meshes,
        "mesh_count": len(meshes),
    }


def ensure_model_cache(cache_dir: Path = DEFAULT_MODEL_CACHE) -> dict[str, Any]:
    """Repair an incomplete bucket download and atomically mark it complete."""
    cache = Path(cache_dir)
    marker = cache / COMPLETION_MARKER
    if marker.is_file():
        try:
            metadata = validate_model_cache(cache)
            marker_payload = json.loads(marker.read_text())
            if marker_payload.get("mesh_count") == EXPECTED_STL_COUNT:
                return metadata
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    cache.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import sync_bucket

    sync_bucket(BUCKET_URI, str(cache), quiet=True)
    metadata = validate_model_cache(cache)
    temporary_marker = cache / f"{COMPLETION_MARKER}.{os.getpid()}.tmp"
    temporary_marker.write_text(json.dumps({"source": BUCKET_URI, **metadata}, indent=2) + "\n")
    temporary_marker.replace(marker)
    return metadata


def _float_vector(element: ET.Element | None, attribute: str, default: str) -> tuple[float, ...]:
    value = element.attrib.get(attribute, default) if element is not None else default
    return tuple(float(item) for item in value.split())


def _joint_signature(element: ET.Element) -> dict[str, Any]:
    origin = element.find("origin")
    axis = element.find("axis")
    limit = element.find("limit")
    return {
        "type": element.attrib.get("type", "fixed"),
        "parent": element.find("parent").attrib["link"],
        "child": element.find("child").attrib["link"],
        "origin_xyz": _float_vector(origin, "xyz", "0 0 0"),
        "origin_rpy": _float_vector(origin, "rpy", "0 0 0"),
        "axis": _float_vector(axis, "xyz", "1 0 0"),
        "lower": float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
        "upper": float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
    }


def verify_kinematic_urdf(
    kinematic_urdf: Path,
    visual_urdf: Path,
    *,
    tolerance: float = 1e-9,
) -> None:
    """Verify joint names, frames, axes, and limits against the visual URDF."""
    local = {element.attrib["name"]: _joint_signature(element) for element in ET.parse(kinematic_urdf).getroot().findall("joint")}
    official = {element.attrib["name"]: _joint_signature(element) for element in ET.parse(visual_urdf).getroot().findall("joint")}
    if set(local) != set(official):
        raise ValueError(
            f"Kinematic and official URDF joint names differ: local={sorted(local)}, official={sorted(official)}"
        )
    for name in local:
        for field in ("type", "parent", "child"):
            if local[name][field] != official[name][field]:
                raise ValueError(f"URDF mismatch for {name}.{field}")
        for field in ("origin_xyz", "origin_rpy", "axis"):
            if any(abs(a - b) > tolerance for a, b in zip(local[name][field], official[name][field], strict=True)):
                raise ValueError(f"URDF mismatch for {name}.{field}")
        for field in ("lower", "upper"):
            a, b = local[name][field], official[name][field]
            if (a is None) != (b is None) or (a is not None and not math.isclose(a, b, abs_tol=tolerance)):
                raise ValueError(f"URDF mismatch for {name}.{field}")


def lerobot_to_urdf_radians(joints: dict[str, float], visual_urdf: Path) -> dict[str, float]:
    """Convert arm degrees and gripper 0..100 into URDF joint radians."""
    root = ET.parse(visual_urdf).getroot()
    gripper = next(element for element in root.findall("joint") if element.attrib.get("name") == "gripper")
    limit = gripper.find("limit")
    if limit is None:
        raise ValueError("Official gripper joint has no limits")
    lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
    converted: dict[str, float] = {}
    for name, value in joints.items():
        if name == "gripper":
            converted[name] = lower + float(np_clip(value, 0.0, 100.0)) / 100.0 * (upper - lower)
        else:
            converted[name] = math.radians(float(value))
    return converted


def np_clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
