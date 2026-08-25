import json
import tempfile
import unittest
from pathlib import Path

from so101.phone_teleop.flight_recorder import FlightRecorder


class FlightRecorderTest(unittest.TestCase):
    def test_records_requested_sent_and_cartesian_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightRecorder(Path(directory), ["shoulder_pan", "gripper"], fps=30)
            sample = recorder.record(
                observation={"shoulder_pan.pos": 1.0, "gripper.pos": 2.0},
                requested_action={"shoulder_pan.pos": 8.0, "gripper.pos": 9.0},
                action={"shoulder_pan.pos": 4.0, "gripper.pos": 5.0},
                cartesian={"error_m": 0.01},
            )
            recorder.close()

            self.assertEqual(sample["requested_commands"]["shoulder_pan"], 8.0)
            self.assertEqual(sample["commands"]["shoulder_pan"], 4.0)
            self.assertEqual(sample["cartesian"]["error_m"], 0.01)
            persisted = json.loads(recorder.session_path.read_text().strip())
            self.assertEqual(persisted["cartesian"], {"error_m": 0.01})


if __name__ == "__main__":
    unittest.main()
