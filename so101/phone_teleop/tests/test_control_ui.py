import threading
import unittest

from fastapi.testclient import TestClient

from so101.phone_teleop.control_ui import RuntimeControlState, create_app


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
        self.assertEqual(payload["phase"], "startup")

    def test_dashboard_assets_are_served(self):
        page = self.client.get("/")
        script = self.client.get("/static/app.js")

        self.assertEqual(page.status_code, 200)
        self.assertIn("SO-101 teleoperation console", page.text)
        self.assertEqual(script.status_code, 200)
        self.assertIn("centerCameraOnRobot", script.text)

    def test_motion_settings_endpoint_is_not_exposed(self):
        response = self.client.post("/api/settings", json={})

        self.assertEqual(response.status_code, 404)

    def test_base_return_is_queued_once(self):
        response = self.client.post("/api/return-to-base")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["return_base_pending"])
        self.assertTrue(self.state.consume_base_return())
        self.assertFalse(self.state.consume_base_return())


if __name__ == "__main__":
    unittest.main()
