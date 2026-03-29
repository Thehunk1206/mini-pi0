import unittest

from mini_pi0.dataset.robot_dataset_mapping import build_robot_dataset_mapping


class RobotDatasetMappingTests(unittest.TestCase):
    def test_mapping_contains_expected_sections(self):
        out = build_robot_dataset_mapping(version="v1.5", include_lerobot=True)
        self.assertIn("local_robosuite", out)
        self.assertIn("hf_datasets", out)
        self.assertIn("robomimic", out["hf_datasets"])
        self.assertIn("lerobot_equivalents", out["hf_datasets"])

    def test_robomimic_matrix_has_known_entries(self):
        out = build_robot_dataset_mapping(version="v1.5", include_lerobot=False)
        rows = out["hf_datasets"]["robomimic"]
        self.assertTrue(any(r["task"] == "lift" and r["dataset_type"] == "ph" for r in rows))
        self.assertTrue(any(r["task"] == "transport" and r["action_dim"] == 14 for r in rows))
        self.assertTrue(any((r["task"] == "transport") and (not r["compatible_now"]) for r in rows))


if __name__ == "__main__":
    unittest.main()
