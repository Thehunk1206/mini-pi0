"""Shared minimal URDF kinematics for desktop and Rerun renderers."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _vector(value: str | None, *, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=float)
    values = [float(part) for part in value.split()]
    if len(values) != 3:
        raise ValueError(f"Expected a three-component URDF vector, got: {value}")
    return np.asarray(values, dtype=float)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm == 0:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ],
        ],
        dtype=float,
    )


def _transform(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


@dataclass(frozen=True)
class URDFJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class URDFVisual:
    """One mesh instance positioned in its owning URDF link frame."""

    link: str
    mesh_path: Path
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    scale: np.ndarray
    rgba: tuple[int, int, int, int]

    @property
    def origin_transform(self) -> np.ndarray:
        transform = _transform(self.origin_xyz, _rpy_matrix(self.origin_rpy))
        transform[:3, :3] = transform[:3, :3] @ np.diag(self.scale)
        return transform


class URDFKinematicModel:
    """Parse a URDF tree and calculate every link-frame transform."""

    def __init__(
        self,
        robot_name: str,
        root_link: str,
        joints: list[URDFJoint],
        visuals: list[URDFVisual] | None = None,
    ) -> None:
        self.robot_name = robot_name
        self.root_link = root_link
        self.joints = joints
        self.visuals = list(visuals or [])
        self.edges = [
            {"joint": joint.name, "parent": joint.parent, "child": joint.child}
            for joint in joints
        ]

    @classmethod
    def from_file(cls, path: Path) -> "URDFKinematicModel":
        path = Path(path)
        root = ET.parse(path).getroot()
        if root.tag != "robot":
            raise ValueError(f"Expected a URDF robot root in {path}")

        links = {element.attrib["name"] for element in root.findall("link")}
        parsed_joints: list[URDFJoint] = []
        child_links: set[str] = set()
        for element in root.findall("joint"):
            parent_element = element.find("parent")
            child_element = element.find("child")
            if parent_element is None or child_element is None:
                raise ValueError(f"URDF joint has no parent or child: {element.attrib}")
            origin_element = element.find("origin")
            axis_element = element.find("axis")
            limit_element = element.find("limit")
            parent = parent_element.attrib["link"]
            child = child_element.attrib["link"]
            child_links.add(child)
            parsed_joints.append(
                URDFJoint(
                    name=element.attrib["name"],
                    joint_type=element.attrib.get("type", "fixed"),
                    parent=parent,
                    child=child,
                    origin_xyz=_vector(
                        origin_element.attrib.get("xyz") if origin_element is not None else None,
                        default=(0.0, 0.0, 0.0),
                    ),
                    origin_rpy=_vector(
                        origin_element.attrib.get("rpy") if origin_element is not None else None,
                        default=(0.0, 0.0, 0.0),
                    ),
                    axis=_vector(
                        axis_element.attrib.get("xyz") if axis_element is not None else None,
                        default=(1.0, 0.0, 0.0),
                    ),
                    lower=(
                        float(limit_element.attrib["lower"])
                        if limit_element is not None and "lower" in limit_element.attrib
                        else None
                    ),
                    upper=(
                        float(limit_element.attrib["upper"])
                        if limit_element is not None and "upper" in limit_element.attrib
                        else None
                    ),
                )
            )

        root_links = links - child_links
        if len(root_links) != 1:
            raise ValueError(f"URDF must have exactly one root link, found: {sorted(root_links)}")
        root_link = next(iter(root_links))

        children: dict[str, list[URDFJoint]] = defaultdict(list)
        for joint in parsed_joints:
            children[joint.parent].append(joint)

        ordered: list[URDFJoint] = []
        pending = deque([root_link])
        while pending:
            parent = pending.popleft()
            for joint in children[parent]:
                ordered.append(joint)
                pending.append(joint.child)

        if len(ordered) != len(parsed_joints):
            raise ValueError("URDF contains disconnected joints or a kinematic cycle")

        material_colors: dict[str, tuple[int, int, int, int]] = {}
        for material in root.findall("material"):
            color = material.find("color")
            if color is not None and "rgba" in color.attrib:
                material_colors[material.attrib["name"]] = _rgba8(color.attrib["rgba"])

        visuals: list[URDFVisual] = []
        for link in root.findall("link"):
            for visual in link.findall("visual"):
                mesh = visual.find("geometry/mesh")
                if mesh is None or "filename" not in mesh.attrib:
                    continue
                origin = visual.find("origin")
                material = visual.find("material")
                rgba = (190, 190, 190, 255)
                if material is not None:
                    color = material.find("color")
                    if color is not None and "rgba" in color.attrib:
                        rgba = _rgba8(color.attrib["rgba"])
                    elif material.attrib.get("name") in material_colors:
                        rgba = material_colors[material.attrib["name"]]
                mesh_path = (path.parent / mesh.attrib["filename"]).resolve()
                visuals.append(
                    URDFVisual(
                        link=link.attrib["name"],
                        mesh_path=mesh_path,
                        origin_xyz=_vector(
                            origin.attrib.get("xyz") if origin is not None else None,
                            default=(0.0, 0.0, 0.0),
                        ),
                        origin_rpy=_vector(
                            origin.attrib.get("rpy") if origin is not None else None,
                            default=(0.0, 0.0, 0.0),
                        ),
                        scale=_vector(
                            mesh.attrib.get("scale"), default=(1.0, 1.0, 1.0)
                        ),
                        rgba=rgba,
                    )
                )
        return cls(root.attrib.get("name", path.stem), root_link, ordered, visuals)

    def revolute_limits_degrees(self) -> dict[str, tuple[float, float]]:
        """Return finite URDF revolute limits converted from radians to degrees."""
        return {
            joint.name: (math.degrees(joint.lower), math.degrees(joint.upper))
            for joint in self.joints
            if joint.joint_type == "revolute"
            and joint.lower is not None
            and joint.upper is not None
        }

    def link_transforms(self, joint_positions_deg: dict[str, float]) -> dict[str, np.ndarray]:
        transforms = {self.root_link: np.eye(4, dtype=float)}
        for joint in self.joints:
            parent_transform = transforms[joint.parent]
            origin = _transform(joint.origin_xyz, _rpy_matrix(joint.origin_rpy))
            motion = np.eye(4, dtype=float)
            if joint.joint_type in {"revolute", "continuous"}:
                angle = math.radians(float(joint_positions_deg.get(joint.name, 0.0)))
                motion[:3, :3] = _axis_angle_matrix(joint.axis, angle)
            elif joint.joint_type == "prismatic":
                distance = float(joint_positions_deg.get(joint.name, 0.0))
                motion[:3, 3] = joint.axis * distance
            elif joint.joint_type != "fixed":
                raise ValueError(f"Unsupported URDF joint type: {joint.joint_type}")
            transforms[joint.child] = parent_transform @ origin @ motion
        return transforms

    def link_positions(
        self, joint_positions_deg: dict[str, float]
    ) -> dict[str, list[float]]:
        return {
            link: transform[:3, 3].astype(float).tolist()
            for link, transform in self.link_transforms(joint_positions_deg).items()
        }

    def lerobot_link_transforms(
        self, joint_positions: dict[str, float]
    ) -> dict[str, np.ndarray]:
        """Transform LeRobot degrees plus gripper 0..100 into URDF link poses."""
        converted = {name: float(value) for name, value in joint_positions.items()}
        gripper_joint = next(
            (joint for joint in self.joints if joint.name == "gripper"), None
        )
        if (
            gripper_joint is not None
            and gripper_joint.lower is not None
            and gripper_joint.upper is not None
            and "gripper" in converted
        ):
            fraction = float(np.clip(converted["gripper"], 0.0, 100.0)) / 100.0
            gripper_radians = gripper_joint.lower + fraction * (
                gripper_joint.upper - gripper_joint.lower
            )
            converted["gripper"] = math.degrees(gripper_radians)
        return self.link_transforms(converted)

    def lerobot_link_positions(
        self, joint_positions: dict[str, float]
    ) -> dict[str, list[float]]:
        return {
            link: transform[:3, 3].astype(float).tolist()
            for link, transform in self.lerobot_link_transforms(joint_positions).items()
        }


def _rgba8(value: str) -> tuple[int, int, int, int]:
    components = [float(part) for part in value.split()]
    if len(components) != 4:
        raise ValueError(f"Expected a four-component URDF color, got: {value}")
    return tuple(
        int(round(255.0 * float(np.clip(component, 0.0, 1.0))))
        for component in components
    )
