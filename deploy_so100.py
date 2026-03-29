import time
from collections import deque

import numpy as np
import torch


class SO100Interface:
    ADDR_GOAL = 116
    ADDR_POS = 132
    BAUD = 1_000_000

    def __init__(self, port="/dev/ttyUSB0", camera_index=0):
        try:
            # Some installations expose this module path.
            from feetech_servo_sdk import PacketHandler, PortHandler
        except Exception:
            # feetech-servo-sdk commonly installs `scservo_sdk`.
            from scservo_sdk import PacketHandler, PortHandler

        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(1.0)
        self.port_handler.openPort()
        self.port_handler.setBaudRate(self.BAUD)

        self.ids = [1, 2, 3, 4, 5, 6]

        import cv2

        self.cv2 = cv2
        self.cam = cv2.VideoCapture(camera_index)

    def _read_joints(self):
        vals = []
        for sid in self.ids:
            raw = self.packet_handler.read4ByteTxRx(self.port_handler, sid, self.ADDR_POS)[0]
            vals.append(raw * 360.0 / 4096.0)
        return np.asarray(vals, dtype=np.float32)

    def _fk(self, joints_deg):
        from roboticstoolbox import DHRobot
        from roboticstoolbox import RevoluteDH as R

        arm = DHRobot(
            [
                R(0.10, 0, 1.5708),
                R(0, 0.12, 0),
                R(0, 0.10, 0),
                R(0, 0, 1.5708),
                R(0.08, 0, -1.5708),
                R(0.05, 0, 0),
            ]
        )
        t = arm.fkine(np.deg2rad(joints_deg))

        try:
            uq = t.UnitQuaternion()
            quat = np.array([uq.s, *uq.v], dtype=np.float32)
        except AttributeError:
            from scipy.spatial.transform import Rotation

            quat_xyzw = Rotation.from_matrix(np.asarray(t.R)).as_quat()
            quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

        return np.asarray(t.t, dtype=np.float32), quat

    def get_obs(self):
        joints = self._read_joints()
        eef_pos, eef_quat = self._fk(joints)

        ok, frame = self.cam.read()
        if not ok:
            raise RuntimeError("Failed to read camera frame")

        frame = self.cv2.resize(frame, (84, 84))
        frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

        return {
            "agentview_image": frame,
            "robot0_eef_pos": eef_pos,
            "robot0_eef_quat": eef_quat,
            "robot0_gripper_qpos": np.array([joints[5] / 180.0, 0.0], dtype=np.float32),
            "robot0_joint_pos": np.deg2rad(joints),
        }

    def send_action(self, action_7d):
        current = self._read_joints()
        # Simple delta-to-joint mapping placeholder.
        target = np.clip(current + action_7d[:6] * 5.0, 0, 180)
        for i, sid in enumerate(self.ids):
            goal = int(target[i] * 4096 / 360)
            self.packet_handler.write4ByteTxRx(self.port_handler, sid, self.ADDR_GOAL, goal)


@torch.no_grad()
def deploy(model, processor, port="/dev/ttyUSB0", execute_steps=4, n_flow_steps=10, camera_index=0):
    robot = SO100Interface(port=port, camera_index=camera_index)
    action_buffer = deque()

    print("Running policy. Ctrl+C to stop.")
    try:
        while True:
            obs = robot.get_obs()
            if not action_buffer:
                img, prop = processor.obs_to_tensors(obs)
                chunk = model.sample(img, prop, n_steps=n_flow_steps)
                chunk = processor.denormalize(chunk.squeeze(0)).cpu().numpy()
                action_buffer.extend(chunk[:execute_steps])

            robot.send_action(action_buffer.popleft())
            time.sleep(1 / 20.0)
    except KeyboardInterrupt:
        print("Stopped.")
