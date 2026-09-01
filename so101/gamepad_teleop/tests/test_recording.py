import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from so101.gamepad_teleop.gamepad import GamepadSample
from so101.gamepad_teleop.record import build_parser
from so101.gamepad_teleop.recording import (
    CameraRig,
    CameraSpec,
    GamepadDatasetRecorder,
    RecordingConfig,
    build_dataset_features,
    build_observation_values,
    build_recording_blueprint,
    parse_camera_spec,
    parse_camera_specs,
    recording_controls_markdown,
    recording_status_markdown,
)
from so101.gamepad_teleop.replay import position_channels, to_hwc_uint8
from so101.teleop.flight_recorder import ElectricalTelemetrySampler
from so101.teleop.runtime import return_to_base


JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def sample(**buttons) -> GamepadSample:
    return GamepadSample(
        timestamp_s=1.0,
        controller_name="test",
        left_x=0.0,
        left_y=0.0,
        right_x=0.0,
        right_y=0.0,
        gripper_direction=0,
        **buttons,
    )


def config(root: Path, *, use_videos: bool = False) -> RecordingConfig:
    return RecordingConfig(
        repo_id="local/test-gamepad-dataset",
        root=root,
        task="pick up the cube",
        camera_specs=(CameraSpec("wrist", 0, 180),),
        use_videos=use_videos,
        image_writer_threads=0,
    )


class FakeMeta:
    total_episodes = 0


class FakeDataset:
    def __init__(self):
        self.meta = FakeMeta()
        self.frames = []
        self.saved = []
        self.clear_count = 0
        self.finalized = False

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self, parallel_encoding=True):
        self.saved.append(list(self.frames))
        self.frames.clear()
        self.meta.total_episodes += 1

    def clear_episode_buffer(self, delete_images=True):
        self.frames.clear()
        self.clear_count += 1

    def finalize(self):
        self.finalized = True


class CameraSpecTest(unittest.TestCase):
    def test_rerun_recording_help_lists_all_gamepad_commands(self):
        help_text = recording_controls_markdown()
        for command in ("A", "Y", "X / Back", "B", "Start / Menu"):
            self.assertIn(f"**{command}**", help_text)

    def test_rerun_status_keeps_state_event_path_and_commands(self):
        status = recording_status_markdown(
            {
                "state": "waiting",
                "episode_index": 2,
                "saved_episodes": 2,
                "episode_frames": 0,
                "episode_seconds": 0.0,
                "last_event": "episode discarded",
                "repo_id": "local/example",
                "task": "pick cube",
                "dataset_root": "/tmp/example",
            }
        )
        for expected in (
            "`waiting`",
            "`episode discarded`",
            "`/tmp/example`",
            "`local/example`",
            "**A** Start episode",
        ):
            self.assertIn(expected, status)

    def test_append_and_overwrite_cli_modes_are_mutually_exclusive(self):
        parser = build_parser()
        appended = parser.parse_args(["--task", "test", "--append"])
        self.assertTrue(appended.resume)
        self.assertFalse(appended.overwrite)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--task", "test", "--append", "--overwrite"]
            )

    def test_named_camera_id_and_rotation_are_parsed(self):
        self.assertEqual(
            parse_camera_spec("wrist=0:180"),
            CameraSpec("wrist", 0, 180),
        )
        self.assertEqual(
            parse_camera_spec("overview=/dev/video4:90"),
            CameraSpec("overview", Path("/dev/video4"), 90),
        )

    def test_default_is_upside_down_wrist_camera_and_duplicates_are_rejected(self):
        self.assertEqual(parse_camera_specs(None), (CameraSpec("wrist", 0, 180),))
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            parse_camera_specs(("wrist=0", "wrist=1"))
        with self.assertRaisesRegex(ValueError, "source"):
            parse_camera_specs(("wrist=0", "room=0"))

    def test_camera_rig_supports_multiple_frames(self):
        class Camera:
            def __init__(self, frame):
                self.frame = frame
                self.is_connected = False

            def connect(self):
                self.is_connected = True

            def read_latest(self, max_age_ms=500):
                return self.frame

            def disconnect(self):
                self.is_connected = False

        frames = {
            "wrist": np.zeros((6, 8, 3), dtype=np.uint8),
            "room": np.zeros((8, 6, 3), dtype=np.uint8),
        }
        rig = CameraRig(
            (CameraSpec("wrist", 0, 180), CameraSpec("room", 1, 90)),
            fps=30,
            width=8,
            height=6,
            camera_factory=lambda spec, *_: Camera(frames[spec.name]),
        )
        rig.connect()
        self.assertTrue(rig.is_connected)
        self.assertEqual(rig.frame_shapes, {"wrist": (6, 8, 3), "room": (8, 6, 3)})
        self.assertEqual(set(rig.read()), {"wrist", "room"})
        rig.disconnect()
        self.assertFalse(rig.is_connected)


class DatasetRecordingTest(unittest.TestCase):
    def test_replay_selects_only_matching_named_position_channels(self):
        features = build_dataset_features(
            JOINTS,
            {"wrist": (6, 8, 3)},
            use_videos=True,
        )
        channels = position_channels(features)
        self.assertEqual(channels.joint_names, JOINTS)
        self.assertEqual(channels.action_indices, tuple(range(6)))
        self.assertEqual(channels.state_indices, tuple(range(6)))

    def test_replay_converts_chw_float_camera_frames_to_hwc_uint8(self):
        frame = np.full((3, 6, 8), 0.5, dtype=np.float32)
        converted = to_hwc_uint8(frame)
        self.assertEqual(converted.shape, (6, 8, 3))
        self.assertEqual(converted.dtype, np.uint8)
        self.assertEqual(int(converted[0, 0, 0]), 127)

    def test_existing_root_requires_explicit_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            recorder = GamepadDatasetRecorder(
                config(root),
                JOINTS,
                {"wrist": (6, 8, 3)},
            )
            with self.assertRaisesRegex(FileExistsError, "--append.*--overwrite"):
                recorder.open()
            recorder.finalize()

    def test_overwrite_archives_existing_root_before_starting_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            (root / "old-recording.txt").write_text("recoverable")
            recorder = GamepadDatasetRecorder(
                replace(config(root), overwrite=True),
                JOINTS,
                {"wrist": (6, 8, 3)},
            )
            recorder.open()
            self.assertTrue(root.exists())
            self.assertIsNotNone(recorder.backup_path)
            assert recorder.backup_path is not None
            self.assertEqual(
                (recorder.backup_path / "old-recording.txt").read_text(),
                "recoverable",
            )
            recorder.finalize()

    def test_base_return_frames_remain_inside_active_episode(self):
        class Robot:
            bus = object()

            def __init__(self):
                self.positions = {joint: 0.0 for joint in JOINTS}

            def get_observation(self):
                return {
                    f"{joint}.pos": value
                    for joint, value in self.positions.items()
                }

            def send_action(self, action):
                self.positions = {
                    joint: float(action[f"{joint}.pos"])
                    for joint in JOINTS
                }
                return dict(action)

        class FlightRecorder:
            def set_phase(self, _phase):
                pass

            def record(self, **_kwargs):
                pass

            def record_electrical_summary(self, _sampler):
                pass

        class Telemetry:
            def maybe_read(self, _bus):
                return {}

        class Snapshot:
            def to_dict(self):
                return {}

        class Visualizer:
            def log(self, _observation, _action):
                return Snapshot(), Snapshot()

        with tempfile.TemporaryDirectory() as directory:
            dataset = FakeDataset()
            recorder = GamepadDatasetRecorder(
                config(Path(directory)),
                JOINTS,
                {"wrist": (6, 8, 3)},
                dataset=dataset,
            )
            recorder.open()
            recorder.start_episode()
            image = {"wrist": np.zeros((6, 8, 3), dtype=np.uint8)}

            def capture(observation, action, electrical):
                recorder.add_frame(
                    observation,
                    action,
                    image,
                )

            with patch("so101.teleop.runtime.precise_sleep"):
                return_to_base(
                    Robot(),
                    {joint: 0.5 for joint in JOINTS},
                    list(JOINTS),
                    FlightRecorder(),
                    Telemetry(),
                    Visualizer(),
                    None,
                    clutch_label=None,
                    frame_callback=capture,
                )

            self.assertEqual(recorder.state, recorder.RECORDING)
            self.assertEqual(recorder.episode_frames, 1)
            self.assertEqual(len(dataset.frames), 1)

    def test_recording_telemetry_reads_joint_velocity_and_effort_registers(self):
        class Bus:
            def __init__(self):
                self.calls = []

            def sync_read(self, register, **_kwargs):
                self.calls.append(register)
                return {joint: 1 for joint in JOINTS}

        bus = Bus()
        sampler = ElectricalTelemetrySampler(
            list(JOINTS),
            frequency_hz=5.0,
            include_velocity=True,
            fast_load=True,
        )
        values = sampler.maybe_read(bus, force=True)
        self.assertEqual(
            set(bus.calls),
            {
                "Present_Voltage",
                "Present_Current",
                "Present_Velocity",
                "Present_Load",
                "Present_Temperature",
            },
        )
        self.assertEqual(values["shoulder_pan"]["velocity_raw"], 1)

    def test_features_include_only_joint_positions_images_and_actions(self):
        features = build_dataset_features(
            JOINTS,
            {"wrist": (6, 8, 3), "room": (6, 8, 3)},
            use_videos=True,
        )
        self.assertEqual(features["action"]["shape"], (6,))
        state_names = features["observation.state"]["names"]
        self.assertEqual(state_names, [f"{joint}.pos" for joint in JOINTS])
        self.assertEqual(features["observation.state"]["shape"], (6,))
        self.assertEqual(features["observation.images.wrist"]["dtype"], "video")
        self.assertEqual(features["observation.images.room"]["shape"], (6, 8, 3))

    def test_observation_values_contain_only_positions_and_cameras(self):
        observation = {f"{joint}.pos": index for index, joint in enumerate(JOINTS)}
        wrist = np.zeros((6, 8, 3), dtype=np.uint8)
        values = build_observation_values(
            JOINTS,
            observation,
            {"wrist": wrist},
        )
        self.assertEqual(
            set(values),
            {"wrist", *(f"{joint}.pos" for joint in JOINTS)},
        )
        self.assertIs(values["wrist"], wrist)

    def test_gamepad_state_machine_saves_only_success_and_discards_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeDataset()
            recorder = GamepadDatasetRecorder(
                config(Path(directory)),
                JOINTS,
                {"wrist": (6, 8, 3)},
                dataset=fake,
            )
            recorder.open()
            recorder.handle_gamepad(sample(start_episode=True))
            self.assertFalse(
                recorder.handle_gamepad(sample(return_to_base=True))
            )
            self.assertEqual(recorder.state, recorder.RECORDING)
            observation = {f"{joint}.pos": 0.0 for joint in JOINTS}
            action = observation.copy()
            image = {"wrist": np.zeros((6, 8, 3), dtype=np.uint8)}
            recorder.add_frame(
                observation,
                action,
                image,
            )
            recorder.handle_gamepad(sample(success=True))
            self.assertEqual(recorder.saved_episodes, 1)
            self.assertEqual(len(fake.saved[0]), 1)

            recorder.handle_gamepad(sample(start_episode=True))
            recorder.add_frame(
                observation,
                action,
                image,
            )
            recorder.handle_gamepad(sample(failure=True))
            self.assertEqual(recorder.saved_episodes, 1)
            self.assertEqual(fake.clear_count, 1)
            self.assertTrue(recorder.handle_gamepad(sample(stop_recording=True)))
            recorder.finalize()
            self.assertTrue(fake.finalized)

    def test_real_lerobot_dataset_round_trip_without_video(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            recorder = GamepadDatasetRecorder(
                config(root),
                JOINTS,
                {"wrist": (6, 8, 3)},
            )
            recorder.open()
            recorder.start_episode()
            observation = {
                f"{joint}.pos": float(index)
                for index, joint in enumerate(JOINTS)
            }
            image = {"wrist": np.full((6, 8, 3), 127, dtype=np.uint8)}
            for _ in range(3):
                recorder.add_frame(
                    observation,
                    observation,
                    image,
                )
            recorder.save_episode()
            recorder.finalize()

            resumed_config = replace(config(root), resume=True)
            resumed = GamepadDatasetRecorder(
                resumed_config,
                JOINTS,
                {"wrist": (6, 8, 3)},
            )
            resumed.open()
            resumed.start_episode()
            resumed.add_frame(
                observation,
                observation,
                image,
            )
            resumed.save_episode()
            resumed.finalize()

            loaded = LeRobotDataset(
                repo_id="local/test-gamepad-dataset",
                root=root,
            )
            self.assertEqual(loaded.num_episodes, 2)
            self.assertEqual(len(loaded), 4)
            first = loaded[0]
            self.assertEqual(tuple(first["observation.state"].shape), (6,))
            self.assertEqual(tuple(first["action"].shape), (6,))
            self.assertEqual(tuple(first["observation.images.wrist"].shape), (3, 6, 8))

    def test_rerun_blueprint_accepts_multiple_camera_views(self):
        blueprint = build_recording_blueprint(("wrist", "room", "side"))
        self.assertIsNotNone(blueprint)


if __name__ == "__main__":
    unittest.main()
