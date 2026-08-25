import threading
import unittest

from fastapi.testclient import TestClient

from so101.phone_teleop.control_ui import (
    ControlSettings,
    RuntimeControlState,
    create_app,
)


INITIAL_SETTINGS = ControlSettings(
    max_relative_target_deg=4.0,
    phone_translation_gain=0.15,
    max_ee_step_m=0.03,
    gripper_speed_factor=8.0,
    servo_acceleration=20,
)


class DesktopControlAPITest(unittest.TestCase):
    def setUp(self):
        self.state = RuntimeControlState(INITIAL_SETTINGS)
        self.client = TestClient(create_app(self.state, threading.Event()))

    def test_state_contains_initial_profile(self):
        response = self.client.get("/api/state")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["desired_settings"]["servo_acceleration"], 20)
        self.assertTrue(payload["settings_pending"])
        self.assertEqual(payload["phase"], "startup")

    def test_dashboard_assets_are_served(self):
        page = self.client.get("/")
        script = self.client.get("/static/app.js")

        self.assertEqual(page.status_code, 200)
        self.assertIn("SO-101 teleoperation console", page.text)
        self.assertEqual(script.status_code, 200)
        self.assertIn("centerCameraOnRobot", script.text)

    def test_settings_are_queued_for_the_control_loop(self):
        updated = {
            "max_relative_target_deg": 5.0,
            "phone_translation_gain": 0.2,
            "max_ee_step_m": 0.04,
            "gripper_speed_factor": 10.0,
            "servo_acceleration": 30,
        }

        response = self.client.post("/api/settings", json=updated)

        self.assertEqual(response.status_code, 200)
        version, settings = self.state.pending_settings(0)
        self.assertEqual(version, 1)
        self.assertEqual(settings, ControlSettings(**updated))

    def test_settings_are_rejected_while_phone_clutch_is_held(self):
        self.state.publish(phone_enabled=True)

        response = self.client.post(
            "/api/settings", json=INITIAL_SETTINGS.__dict__
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("Release Hold to move", response.json()["detail"])

    def test_invalid_motion_profile_is_rejected(self):
        invalid = {**INITIAL_SETTINGS.__dict__, "servo_acceleration": 0}

        response = self.client.post("/api/settings", json=invalid)

        self.assertEqual(response.status_code, 422)

    def test_base_return_is_queued_once(self):
        response = self.client.post("/api/return-to-base")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["return_base_pending"])
        self.assertTrue(self.state.consume_base_return())
        self.assertFalse(self.state.consume_base_return())


if __name__ == "__main__":
    unittest.main()
