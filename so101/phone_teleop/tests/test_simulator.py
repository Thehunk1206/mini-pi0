import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from so101.phone_teleop.simulator import SimulationEngine, create_app


class SimulatorAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SimulationEngine()
        cls.client = TestClient(create_app(cls.engine))

    def setUp(self):
        self.engine.load_scenario("phone_step")

    def test_all_comparison_streams_and_history_are_available(self):
        state = self.client.get("/api/state")
        self.assertEqual(state.status_code, 200)
        self.assertEqual(
            set(state.json()["streams"]),
            {"raw_ik", "one_euro", "kalman", "quintic", "ruckig", "measured"},
        )
        history = self.client.get("/api/history").json()
        self.assertGreater(len(history["samples"]), 100)
        self.assertEqual(len(history["joint_names"]), 6)

    def test_play_pause_restart_step_scrub_and_speed(self):
        self.assertTrue(self.client.post("/api/playback", json={"action": "play", "speed": 2}).json()["playing"])
        paused = self.client.post("/api/playback", json={"action": "pause"}).json()
        self.assertFalse(paused["playing"])
        stepped = self.client.post("/api/playback", json={"action": "step"}).json()
        self.assertGreater(stepped["playback_time_s"], paused["playback_time_s"])
        scrubbed = self.client.post("/api/playback", json={"action": "scrub", "time_s": 3.0}).json()
        self.assertAlmostEqual(scrubbed["playback_time_s"], 3.0)
        restarted = self.client.post("/api/playback", json={"action": "restart"}).json()
        self.assertEqual(restarted["playback_time_s"], 0.0)

    def test_profile_and_filter_settings_are_bounded_and_conflict_during_hold(self):
        accepted = self.client.put("/api/settings", json={"profile": "Balanced", "beta": 4.0})
        self.assertEqual(accepted.status_code, 200)
        self.client.post("/api/playback", json={"action": "scrub", "time_s": 1.0})
        conflict = self.client.put("/api/settings", json={"profile": "Safe"})
        self.assertEqual(conflict.status_code, 409)
        self.client.post("/api/playback", json={"action": "restart"})
        invalid = self.client.put("/api/settings", json={"deadband_m": 0.5})
        self.assertEqual(invalid.status_code, 422)

    def test_remote_clients_are_rejected(self):
        remote = TestClient(create_app(self.engine), client=("192.0.2.10", 50000))
        self.assertEqual(remote.get("/api/state").status_code, 403)

    def test_existing_jsonl_session_can_be_replayed(self):
        recording = Path("logs/phone_teleop/session_20260823_203753.jsonl").resolve()
        if not recording.is_file():
            self.skipTest("phone teleoperation capture is unavailable")
        response = self.client.post(
            "/api/scenario",
            json={"name": "recorded_session", "recording": str(recording)},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["scenario"], "recorded_session")
        self.assertTrue(response.json()["electrical"])


if __name__ == "__main__":
    unittest.main()
