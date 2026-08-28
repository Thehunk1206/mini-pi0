import threading
import unittest

from fastapi.testclient import TestClient

from so101.phone_teleop.control_ui import RuntimeControlState, create_app
from so101.phone_teleop.filtering import DEFAULT_PHONE_FILTER_SETTINGS


class DesktopControlAPITest(unittest.TestCase):
    def setUp(self):
        self.state = RuntimeControlState()
        self.client = TestClient(create_app(self.state, threading.Event()))

    def test_state_contains_initial_runtime_status(self):
        response = self.client.get("/api/state")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["connected"])
        self.assertFalse(payload["phone_enabled"])
        self.assertFalse(payload["control_mapping"]["orientation_enabled"])
        self.assertEqual(payload["phase"], "startup")

    def test_dashboard_assets_are_served(self):
        page = self.client.get("/")
        script = self.client.get("/static/app.js")
        three_core = self.client.get("/static/vendor/three/three.core.min.js")
        urdf_loader = self.client.get("/static/vendor/urdf-loader/URDFLoader.js")

        self.assertEqual(page.status_code, 200)
        self.assertIn("SO-101", page.text)
        self.assertIn("Trajectory Lab", page.text)
        self.assertEqual(script.status_code, 200)
        self.assertIn("URDFLoader", script.text)
        self.assertEqual(three_core.status_code, 200)
        self.assertEqual(urdf_loader.status_code, 200)

    def test_motion_profile_change_is_accepted_only_with_hold_released(self):
        response = self.client.put("/api/settings", json={"profile": "Balanced"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.state.consume_settings(),
            {
                "profile": "Balanced",
                "filter_settings": DEFAULT_PHONE_FILTER_SETTINGS,
            },
        )

        self.state.publish(phone_enabled=True)
        response = self.client.put("/api/settings", json={"profile": "Responsive"})
        self.assertEqual(response.status_code, 409)

    def test_live_filter_settings_are_bounded(self):
        response = self.client.put(
            "/api/settings",
            json={"profile": "Smooth", "min_cutoff_hz": 0.75, "beta": 1.0},
        )
        self.assertEqual(response.status_code, 200)
        requested = self.state.consume_settings()
        self.assertEqual(requested["profile"], "Smooth")
        self.assertEqual(requested["filter_settings"]["min_cutoff_hz"], 0.75)

        invalid = self.client.put("/api/settings", json={"deadband_m": 0.5})
        self.assertEqual(invalid.status_code, 422)

    def test_base_return_is_queued_once(self):
        response = self.client.post("/api/return-to-base")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["return_base_pending"])
        self.assertTrue(self.state.consume_base_return())
        self.assertFalse(self.state.consume_base_return())

    def test_history_can_be_polled_incrementally(self):
        self.state.publish(positions={"shoulder_pan": 1.0})
        self.state.publish(positions={"shoulder_pan": 2.0})

        full = self.client.get("/api/history").json()
        self.assertEqual(full["latest_sequence"], 2)
        self.assertEqual(
            [sample["sequence"] for sample in full["samples"]],
            [1, 2],
        )

        incremental = self.client.get(
            "/api/history", params={"after_sequence": 1}
        ).json()
        self.assertFalse(incremental["reset"])
        self.assertEqual(
            [sample["sequence"] for sample in incremental["samples"]],
            [2],
        )


if __name__ == "__main__":
    unittest.main()
