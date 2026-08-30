import tempfile
import unittest
from pathlib import Path

from so101.phone_teleop.model_assets import (
    EXPECTED_STL_COUNT,
    KINEMATIC_URDF_PATH,
    MODEL_FILENAME,
    lerobot_to_urdf_radians,
    validate_model_cache,
    verify_kinematic_urdf,
)


class OfficialModelAssetTest(unittest.TestCase):
    def test_full_urdf_and_thirteen_referenced_meshes_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            meshes = [f"assets/mesh_{index}.stl" for index in range(EXPECTED_STL_COUNT)]
            (cache / "assets").mkdir()
            visuals = "".join(f'<link name="l{index}"><visual><geometry><mesh filename="{mesh}"/></geometry></visual></link>' for index, mesh in enumerate(meshes))
            (cache / MODEL_FILENAME).write_text(f'<robot name="test">{visuals}</robot>')
            for mesh in meshes:
                (cache / mesh).write_bytes(b"solid mesh\nendsolid mesh\n")
            metadata = validate_model_cache(cache)
            self.assertEqual(metadata["mesh_count"], 13)
            (cache / meshes[-1]).unlink()
            with self.assertRaises(FileNotFoundError):
                validate_model_cache(cache)

    def test_gripper_zero_is_closed_and_hundred_is_open(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf = Path(directory) / MODEL_FILENAME
            urdf.write_text('<robot name="test"><joint name="gripper" type="revolute"><limit lower="-0.2" upper="1.8"/></joint></robot>')
            closed = lerobot_to_urdf_radians({"gripper": 0.0, "shoulder_pan": 180.0}, urdf)
            opened = lerobot_to_urdf_radians({"gripper": 100.0}, urdf)
            self.assertAlmostEqual(closed["gripper"], -0.2)
            self.assertAlmostEqual(opened["gripper"], 1.8)
            self.assertAlmostEqual(closed["shoulder_pan"], 3.141592653589793)

    def test_checked_in_kinematics_matches_cached_official_model_when_available(self):
        official = Path.home() / ".cache/huggingface/lerobot/robot-urdfs/so101" / MODEL_FILENAME
        if not official.is_file():
            self.skipTest("official model cache is not populated")
        verify_kinematic_urdf(KINEMATIC_URDF_PATH, official)


if __name__ == "__main__":
    unittest.main()
