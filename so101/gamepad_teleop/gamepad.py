"""Safe, testable gamepad input and articulated target integration.

The SDL/pygame reader is deliberately kept separate from the pure motion math.
This makes shaping and target integration testable without a controller or a
servo bus.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from lerobot.model.kinematics import RobotKinematics


class GamepadConnectionError(RuntimeError):
    """Raised when SDL cannot provide the configured controller."""


@dataclass(frozen=True)
class XboxLayout:
    """SDL GameController's standardized Xbox axis/button indices."""

    left_x_axis: int = 0
    left_y_axis: int = 1
    right_x_axis: int = 2
    right_y_axis: int = 3
    left_trigger_axis: int = 4
    right_trigger_axis: int = 5
    failure_button: int = 2  # X
    success_button: int = 3  # Y
    dpad_up_button: int = 11
    dpad_down_button: int = 12
    rerecord_button: int = 4  # Back/View
    base_button: int = 1  # B
    record_button: int = 0  # A
    stop_recording_button: int = 6  # Start/Menu


TRIGGER_ACTIVE_THRESHOLD = 0.12
SHOULDER_ORIGIN_M = np.asarray([0.0388353, 0.0, 0.0624], dtype=float)
SHOULDER_LIFT_FRAME = "upper_arm_link"
FEEDBACK_REARM_INTERVAL_S = 0.5
XBOX_360_VENDOR_ID = 0x045E
XBOX_360_PRODUCT_ID = 0x028E
XBOX_360_GUID_ID = "5e0400008e020000"
PAN_CONTROL_MODES = ("velocity", "absolute")


def find_elbow_singularity_deg(
    kinematics: RobotKinematics,
    joint_names: list[str] | tuple[str, ...],
    joint_limits_deg: dict[str, tuple[float, float]],
) -> float:
    """Find maximum wrist-pivot reach inside the calibrated elbow interval.

    LeRobot motor degrees are calibrated coordinates: elbow ``0`` is not the
    straight-arm configuration. Deriving the maximum from FK avoids embedding
    that incorrect assumption in the branch guard.
    """

    names = list(joint_names)
    elbow_index = names.index("elbow_flex")
    lower, upper = joint_limits_deg["elbow_flex"]
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("invalid calibrated elbow interval")
    seed = np.zeros(len(names), dtype=float)

    def wrist_radius(elbow_deg: float) -> float:
        joints = seed.copy()
        joints[elbow_index] = elbow_deg
        wrist = np.asarray(
            kinematics.forward_kinematics(joints)[:3, 3], dtype=float
        )
        shoulder = np.asarray(
            kinematics.robot.get_T_world_frame(SHOULDER_LIFT_FRAME)[:3, 3],
            dtype=float,
        )
        return float(np.linalg.norm(wrist - shoulder))

    # Bracket the global maximum before refining it. The calibrated interval is
    # small, so this startup-only search is inexpensive and deterministic.
    samples = np.linspace(lower, upper, 257)
    radii = np.asarray([wrist_radius(float(value)) for value in samples])
    best = int(np.argmax(radii))
    if best in (0, len(samples) - 1):
        raise ValueError(
            "straight-elbow singularity is outside the calibrated elbow interval"
        )
    left = float(samples[best - 1])
    right = float(samples[best + 1])
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    x1 = right - (right - left) / golden_ratio
    x2 = left + (right - left) / golden_ratio
    f1 = wrist_radius(x1)
    f2 = wrist_radius(x2)
    for _ in range(32):
        if f1 < f2:
            left = x1
            x1, f1 = x2, f2
            x2 = left + (right - left) / golden_ratio
            f2 = wrist_radius(x2)
        else:
            right = x2
            x2, f2 = x1, f1
            x1 = right - (right - left) / golden_ratio
            f1 = wrist_radius(x1)
    return 0.5 * (left + right)


@dataclass(frozen=True)
class GamepadMotionSettings:
    """Commissioning values for joystick-style planar arm control."""

    deadzone: float = 0.12
    expo: float = 0.65
    axis_slew_rate_per_s: float = 4.0
    pan_control_mode: str = "velocity"
    planar_velocity_m_s: float = 0.12
    height_velocity_m_s: float = 0.12
    shoulder_pan_velocity_deg_s: float = 45.0
    wrist_flex_velocity_deg_s: float = 50.0
    wrist_roll_velocity_deg_s: float = 70.0
    planar_offset_limits_m: tuple[float, float] = (-0.10, 0.40)
    height_offset_limits_m: tuple[float, float] = (-0.12, 0.15)
    extended_elbow_stop_deg: float = 2.0
    minimum_dt_s: float = 1.0 / 120.0
    maximum_dt_s: float = 0.1
    reset_gap_s: float = 0.2

    def __post_init__(self) -> None:
        if self.pan_control_mode not in PAN_CONTROL_MODES:
            raise ValueError(
                f"pan_control_mode must be one of {PAN_CONTROL_MODES}"
            )
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError("deadzone must be in [0, 1)")
        if not 0.0 <= self.expo <= 1.0:
            raise ValueError("expo must be in [0, 1]")
        speeds = (
            self.axis_slew_rate_per_s,
            self.planar_velocity_m_s,
            self.height_velocity_m_s,
            self.shoulder_pan_velocity_deg_s,
            self.wrist_flex_velocity_deg_s,
            self.wrist_roll_velocity_deg_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in speeds):
            raise ValueError("gamepad velocity limits must be finite and positive")
        if (
            not math.isfinite(self.extended_elbow_stop_deg)
            or self.extended_elbow_stop_deg <= 0.0
        ):
            raise ValueError("extended-elbow stop must be finite and positive")
        for name, bounds in (
            ("planar", self.planar_offset_limits_m),
            ("height", self.height_offset_limits_m),
        ):
            if (
                len(bounds) != 2
                or not all(math.isfinite(value) for value in bounds)
                or bounds[0] >= bounds[1]
                or not bounds[0] <= 0.0 <= bounds[1]
            ):
                raise ValueError(f"invalid {name} offset limits")
        if not 0.0 < self.minimum_dt_s <= self.maximum_dt_s:
            raise ValueError("invalid gamepad timestep range")
        if self.reset_gap_s <= self.maximum_dt_s:
            raise ValueError("reset gap must exceed maximum timestep")


@dataclass(frozen=True)
class GamepadSample:
    """One polled controller sample in standardized Xbox semantics."""

    timestamp_s: float
    controller_name: str
    left_x: float
    left_y: float
    right_x: float
    right_y: float
    gripper_direction: int
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    dpad_vertical: int = 0
    success: bool = False
    failure: bool = False
    rerecord: bool = False
    return_to_base: bool = False
    start_episode: bool = False
    stop_recording: bool = False
    raw_axes: tuple[float, ...] = field(default_factory=tuple)
    raw_buttons: tuple[bool, ...] = field(default_factory=tuple)

    @property
    def episode_event(self) -> str | None:
        if self.success:
            return "success"
        if self.failure:
            return "failure"
        if self.rerecord:
            return "rerecord"
        if self.start_episode:
            return "start"
        if self.stop_recording:
            return "finish"
        return None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["episode_event"] = self.episode_event
        return payload


def shape_axis(value: float, *, deadzone: float, expo: float) -> float:
    """Remove and rescale a radial-free scalar deadzone, then apply cubic expo.

    Rescaling preserves the full output range: the deadzone boundary maps to
    zero and a fully deflected stick still maps to exactly +/-1.
    """

    value = float(np.clip(value, -1.0, 1.0))
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    normalized = (magnitude - deadzone) / (1.0 - deadzone)
    shaped = (1.0 - expo) * normalized + expo * normalized**3
    return math.copysign(shaped, value)


class PygameGamepad:
    """Poll one SDL gamepad without ever opening the robot or servo bus."""

    def __init__(self, controller_index: int = 0, layout: XboxLayout | None = None) -> None:
        self.controller_index = int(controller_index)
        self.layout = layout or XboxLayout()
        self._pygame = None
        self._controller_module = None
        self._controller = None
        self._direct_hid_path: bytes | None = None
        self._direct_hid_lock = threading.Lock()
        self._direct_hid_stop_timer: threading.Timer | None = None
        self._direct_hid_open_error: str | None = None
        self._previous_buttons: tuple[bool, ...] = ()
        self._last_feedback_s = float("-inf")
        self._active_feedback_event: str | None = None
        self._feedback_clear_since_s: float | None = None
        self.last_rumble_result: dict[str, Any] = {}
        self.guid = ""
        self.name = ""

    @property
    def is_connected(self) -> bool:
        return bool(
            self._controller is not None
            and self._controller.get_init()
            and self._controller.attached()
        )

    def connect(self) -> None:
        # Pygame's full init also initializes audio and can block on macOS.
        # Keep a native hidden video backend: SDL's dummy backend can enumerate
        # a macOS controller but then leave its axes/buttons frozen at the
        # initial HID snapshot. Rerun becomes the foreground application, so
        # background joystick updates must also be enabled before SDL starts.
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        # Keep this wired Xbox controller on SDL's HIDAPI input and rumble
        # path. The controller is not enumerated through pygame's Apple/MFI
        # backend on this machine.
        os.environ.setdefault("SDL_JOYSTICK_HIDAPI", "1")
        os.environ.setdefault("SDL_JOYSTICK_HIDAPI_XBOX", "1")
        os.environ.setdefault("SDL_JOYSTICK_MFI", "0")
        try:
            import pygame
            from pygame._sdl2 import controller
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise GamepadConnectionError(
                "pygame is missing; install so101/gamepad_teleop/requirements.txt"
            ) from exc

        self._pygame = pygame
        self._controller_module = controller
        pygame.display.init()
        pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
        controller.init()
        count = controller.get_count()
        if not 0 <= self.controller_index < count:
            self.disconnect()
            raise GamepadConnectionError(
                f"Controller index {self.controller_index} is unavailable; SDL found {count} controller(s)"
            )
        if not controller.is_controller(self.controller_index):
            self.disconnect()
            raise GamepadConnectionError(
                f"SDL device {self.controller_index} is a joystick but has no "
                "standard GameController mapping"
            )

        game_controller = controller.Controller(self.controller_index)
        game_controller.init()
        self._controller = game_controller
        self.name = game_controller.name
        self.guid = game_controller.as_joystick().get_guid()
        self._connect_direct_hid_haptics()

        # SDL can report transient values until the first few HID packets arrive.
        for _ in range(5):
            pygame.event.get()
            time.sleep(0.02)
        self._previous_buttons = self._buttons()

    def _connect_direct_hid_haptics(self) -> None:
        """Locate the normal Xbox output endpoint bypassing SDL's macOS alias.

        Apple's XboxGamepad driver exposes the real 045e:028e controller as a
        version-zero "Steam Virtual Gamepad". SDL therefore disables rumble.
        hidapi needs one report-ID byte prepended to the normal eight-byte XUSB
        command; the Apple driver removes that prefix and forwards the command.

        Do not keep this HID endpoint open. On pygame 2.6/SDL 2.28, a second
        persistent handle starves the SDL controller handle and freezes all
        axes and buttons. Each vibration write opens and immediately closes
        the endpoint instead.
        """

        if sys.platform != "darwin" or XBOX_360_GUID_ID not in self.guid.lower():
            return
        try:
            import hid

            interfaces = hid.enumerate(XBOX_360_VENDOR_ID, XBOX_360_PRODUCT_ID)
            interface = next(
                item
                for item in interfaces
                if item.get("usage_page") == 1 and item.get("usage") == 5
            )
            self._direct_hid_path = interface["path"]
        except Exception as exc:
            self._direct_hid_open_error = str(exc)

    def _write_direct_hid_packet(self, packet: bytes) -> bool:
        """Write one haptic packet without retaining the controller endpoint."""

        if self._direct_hid_path is None:
            return False
        import hid

        device = hid.device()
        try:
            device.open_path(self._direct_hid_path)
            return device.write(packet) == len(packet)
        finally:
            device.close()

    @staticmethod
    def _direct_hid_packet(low: float, high: float) -> bytes:
        return bytes(
            (
                0x00,  # hidapi report-ID prefix
                0x00,
                0x08,
                0x00,
                round(255.0 * low),
                round(255.0 * high),
                0x00,
                0x00,
                0x00,
            )
        )

    def _stop_direct_hid_rumble(self) -> None:
        with self._direct_hid_lock:
            self._write_direct_hid_packet(self._direct_hid_packet(0.0, 0.0))
            self._direct_hid_stop_timer = None

    def _rumble_direct_hid(self, low: float, high: float, duration_ms: int) -> bool:
        if self._direct_hid_path is None:
            return False
        with self._direct_hid_lock:
            if self._direct_hid_stop_timer is not None:
                self._direct_hid_stop_timer.cancel()
                self._direct_hid_stop_timer = None
            packet = self._direct_hid_packet(low, high)
            if not self._write_direct_hid_packet(packet):
                return False
            if duration_ms > 0 and (low > 0.0 or high > 0.0):
                timer = threading.Timer(
                    duration_ms / 1000.0, self._stop_direct_hid_rumble
                )
                timer.daemon = True
                self._direct_hid_stop_timer = timer
                timer.start()
            return True

    def disconnect(self) -> None:
        if self._direct_hid_stop_timer is not None:
            self._direct_hid_stop_timer.cancel()
            self._direct_hid_stop_timer = None
        if self._direct_hid_path is not None:
            self._stop_direct_hid_rumble()
            self._direct_hid_path = None
        if self._controller is not None:
            try:
                self._controller.quit()
            finally:
                self._controller = None
        if self._controller_module is not None:
            self._controller_module.quit()
            self._controller_module = None
        if self._pygame is not None:
            self._pygame.display.quit()
        self._previous_buttons = ()

    def rumble(
        self,
        low_frequency: float,
        high_frequency: float,
        duration_ms: int,
    ) -> bool:
        """Play a bounded SDL rumble effect when the controller supports it."""

        if self._controller is None:
            return False
        low = float(np.clip(low_frequency, 0.0, 1.0))
        high = float(np.clip(high_frequency, 0.0, 1.0))
        duration = max(0, int(duration_ms))
        direct_hid_supported = False
        controller_supported = False
        joystick_supported = False
        errors: list[str] = []
        try:
            direct_hid_supported = self._rumble_direct_hid(low, high, duration)
        except Exception as exc:
            errors.append(f"direct_hid: {exc}")
        if self._direct_hid_open_error:
            errors.append(f"direct_hid_open: {self._direct_hid_open_error}")
        # The direct endpoint is the verified path for this wired Xbox clone.
        # Avoid asking old SDL to open a competing output handle when it worked.
        if not direct_hid_supported:
            try:
                controller_supported = bool(
                    self._controller.rumble(low, high, duration)
                )
            except Exception as exc:
                errors.append(f"controller: {exc}")
            try:
                # Send through both SDL views for controllers without the
                # direct macOS HID workaround.
                joystick = self._controller.as_joystick()
                joystick_supported = bool(joystick.rumble(low, high, duration))
            except Exception as exc:
                errors.append(f"joystick: {exc}")
        self.last_rumble_result = {
            "direct_hid": direct_hid_supported,
            "controller": controller_supported,
            "joystick": joystick_supported,
            "errors": errors,
            "low_frequency": low,
            "high_frequency": high,
            "duration_ms": duration,
        }
        # Rumble is optional in SDL mappings. A controller without haptics must
        # never stop or destabilize teleoperation.
        return direct_hid_supported or controller_supported or joystick_supported

    def safety_feedback(
        self,
        event: str,
        *,
        timestamp_s: float | None = None,
    ) -> bool:
        """Emit distinct, rate-limited haptics for gamepad safety events."""

        patterns = {
            "joint_limit": (0.8, 1.0, 350),
            "ik_jump": (1.0, 1.0, 450),
            "workspace_limit": (1.0, 0.8, 400),
        }
        pattern = patterns.get(event)
        if pattern is None:
            return False
        now = time.monotonic() if timestamp_s is None else float(timestamp_s)
        # One pulse marks entry into an unsafe region. Do not continuously
        # buzz, or switch patterns while the controller remains unsafe.
        if self._active_feedback_event is not None:
            return False
        played = self.rumble(*pattern)
        # Latch even when this controller reports no rumble support; otherwise
        # the backend would be retried on every 30 Hz control cycle.
        self._last_feedback_s = now
        self._active_feedback_event = event
        self._feedback_clear_since_s = None
        return played

    def clear_safety_feedback(self, *, timestamp_s: float | None = None) -> None:
        """Re-arm feedback only after a stable interval of safe motion."""

        if self._active_feedback_event is None:
            self._feedback_clear_since_s = None
            return
        now = time.monotonic() if timestamp_s is None else float(timestamp_s)
        if self._feedback_clear_since_s is None:
            self._feedback_clear_since_s = now
            return
        if now - self._feedback_clear_since_s >= FEEDBACK_REARM_INTERVAL_S:
            self._active_feedback_event = None
            self._feedback_clear_since_s = None

    def _axes(self) -> tuple[float, ...]:
        if self._controller is None:
            raise GamepadConnectionError("Gamepad is not connected")
        # SDL GameController axes are signed int16 values in a fixed semantic
        # order: left X/Y, right X/Y, left trigger, right trigger.
        return tuple(
            float(value) / (32767.0 if value >= 0 else 32768.0)
            for value in (self._controller.get_axis(i) for i in range(6))
        )

    def _buttons(self) -> tuple[bool, ...]:
        if self._controller is None:
            raise GamepadConnectionError("Gamepad is not connected")
        return tuple(bool(self._controller.get_button(i)) for i in range(15))

    def read(self, timestamp_s: float | None = None) -> GamepadSample:
        if self._pygame is None or self._controller is None:
            raise GamepadConnectionError("Gamepad is not connected")
        if not self._controller.attached():
            raise GamepadConnectionError("Gamepad was disconnected")
        try:
            # event.get() both pumps the native event loop and drains its queue,
            # preventing joystick events from accumulating during long sessions.
            self._pygame.event.get()
            axes = self._axes()
            buttons = self._buttons()
        except self._pygame.error as exc:
            raise GamepadConnectionError(f"Controller read failed: {exc}") from exc

        previous = self._previous_buttons

        def pressed(index: int) -> bool:
            return buttons[index]

        def rising(index: int) -> bool:
            return buttons[index] and (not previous or not previous[index])

        # SDL GameController triggers are independent 0..1 axes. Treat them as
        # buttons after a small noise deadzone so the direct gripper behavior
        # remains predictable: LT opens, RT closes, and pressing both holds.
        left_trigger = float(
            np.clip(axes[self.layout.left_trigger_axis], 0.0, 1.0)
        )
        right_trigger = float(
            np.clip(axes[self.layout.right_trigger_axis], 0.0, 1.0)
        )
        open_pressed = left_trigger > TRIGGER_ACTIVE_THRESHOLD
        close_pressed = right_trigger > TRIGGER_ACTIVE_THRESHOLD
        sample = GamepadSample(
            timestamp_s=time.monotonic() if timestamp_s is None else float(timestamp_s),
            controller_name=self.name,
            left_x=axes[self.layout.left_x_axis],
            left_y=axes[self.layout.left_y_axis],
            right_x=axes[self.layout.right_x_axis],
            right_y=axes[self.layout.right_y_axis],
            gripper_direction=int(open_pressed) - int(close_pressed),
            left_trigger=left_trigger,
            right_trigger=right_trigger,
            dpad_vertical=(
                int(pressed(self.layout.dpad_up_button))
                - int(pressed(self.layout.dpad_down_button))
            ),
            success=rising(self.layout.success_button),
            failure=rising(self.layout.failure_button),
            rerecord=rising(self.layout.rerecord_button),
            return_to_base=rising(self.layout.base_button),
            start_episode=rising(self.layout.record_button),
            stop_recording=rising(self.layout.stop_recording_button),
            raw_axes=axes,
            raw_buttons=buttons,
        )
        self._previous_buttons = buttons
        return sample


def test_controller_vibration(controller_index: int = 0) -> bool:
    """Play one strong controller-only test pulse without opening the robot."""

    gamepad = PygameGamepad(controller_index)
    try:
        gamepad.connect()
        print(f"Controller: {gamepad.name}")
        print("Playing a one-second full-strength vibration test...")
        supported = gamepad.rumble(1.0, 1.0, 1000)
        print(
            "SDL vibration result: "
            + ("supported" if supported else "not supported by this controller/driver")
        )
        print(f"Haptic backends: {gamepad.last_rumble_result}")
        # SDL rumble is asynchronous; do not close the controller before the
        # requested test pulse has actually had time to play.
        time.sleep(1.1)
        return supported
    finally:
        gamepad.disconnect()


class GamepadTargetIntegrator:
    """Integrate reference-style planar IK and three direct joint velocities.

    Left vertical changes wrist-pivot reach, D-pad up/down changes its height,
    and position-only IK supplies shoulder lift and elbow flex. Left horizontal
    directly pans the base; right vertical/horizontal directly command wrist
    flex/roll. A bounded input slew smooths starts and stops without a joint
    trajectory generator.
    """

    def __init__(
        self,
        kinematics: RobotKinematics,
        joint_names: list[str] | tuple[str, ...],
        settings: GamepadMotionSettings | None = None,
        *,
        elbow_singularity_deg: float | None = None,
    ) -> None:
        self.kinematics = kinematics
        self.joint_names = list(joint_names)
        self.settings = settings or GamepadMotionSettings()
        self.elbow_singularity_deg = (
            None
            if elbow_singularity_deg is None
            else float(elbow_singularity_deg)
        )
        if self.elbow_singularity_deg is not None and not math.isfinite(
            self.elbow_singularity_deg
        ):
            raise ValueError("elbow singularity must be finite")
        self.translation_offset_m = np.zeros(3, dtype=float)
        self.shoulder_pan_target_deg: float | None = None
        self.wrist_flex_target_deg: float | None = None
        self.wrist_roll_target_deg: float | None = None
        self._reference_xyz_m: np.ndarray | None = None
        self._reference_pan_deg: float | None = None
        self._reference_radius_m: float | None = None
        self._reference_azimuth_rad: float | None = None
        self._pan_to_azimuth_sign = -1.0
        self._elbow_branch_side = 1.0
        self._maximum_wrist_radius_m: float | None = None
        self._active_elbow_ik_limits_deg: tuple[float, float] | None = None
        self._planar_offset_m = 0.0
        self._height_offset_m = 0.0
        self._smoothed_axes = np.zeros(5, dtype=float)
        self._last_timestamp_s: float | None = None
        self._cartesian_state_before_update: tuple[float, float] | None = None
        self.latest: dict[str, Any] = {}

    def reset(self) -> None:
        self.translation_offset_m.fill(0.0)
        self.shoulder_pan_target_deg = None
        self.wrist_flex_target_deg = None
        self.wrist_roll_target_deg = None
        self._reference_xyz_m = None
        self._reference_pan_deg = None
        self._reference_radius_m = None
        self._reference_azimuth_rad = None
        self._pan_to_azimuth_sign = -1.0
        self._elbow_branch_side = 1.0
        self._maximum_wrist_radius_m = None
        self._active_elbow_ik_limits_deg = None
        self._planar_offset_m = 0.0
        self._height_offset_m = 0.0
        self._smoothed_axes.fill(0.0)
        self._last_timestamp_s = None
        self._cartesian_state_before_update = None
        self.latest = {}

    def rollback_latest_cartesian_step(self) -> bool:
        """Discard the latest reach/height increment after an IK branch stop.

        Direct pan and wrist targets intentionally remain advanced. Recomputing
        the Cartesian offset with the new pan target lets those independent
        controls continue while reach/height stays at its last safe value.
        """

        if self._cartesian_state_before_update is None:
            return False
        self._planar_offset_m, self._height_offset_m = (
            self._cartesian_state_before_update
        )
        self._cartesian_state_before_update = None
        self._update_translation_offset()
        self.latest.update(
            {
                "translation_offset_m": self.translation_offset_m.tolist(),
                "x_offset_m": float(self.translation_offset_m[0]),
                "y_offset_m": float(self.translation_offset_m[1]),
                "z_offset_m": float(self.translation_offset_m[2]),
                "planar_offset_m": self._planar_offset_m,
                "height_offset_m": self._height_offset_m,
                "workspace_clamped": True,
                "extension_clamped": True,
                "cartesian_step_rolled_back": True,
            }
        )
        return True

    @property
    def direct_joint_targets(self) -> dict[str, float]:
        """Return joints intentionally controlled outside position-only IK."""

        if (
            self.shoulder_pan_target_deg is None
            or self.wrist_flex_target_deg is None
            or self.wrist_roll_target_deg is None
        ):
            return {}
        return {
            "shoulder_pan": float(self.shoulder_pan_target_deg),
            "wrist_flex": float(self.wrist_flex_target_deg),
            "wrist_roll": float(self.wrist_roll_target_deg),
        }

    def _latch_reference(
        self,
        measured: np.ndarray,
        measured_positions: dict[str, float],
        joint_limits_deg: dict[str, tuple[float, float]],
    ) -> None:
        self.translation_offset_m.fill(0.0)
        self._planar_offset_m = 0.0
        self._height_offset_m = 0.0
        self._reference_xyz_m = np.asarray(
            self.kinematics.forward_kinematics(measured)[:3, 3], dtype=float
        )
        self._reference_pan_deg = float(measured_positions["shoulder_pan"])
        shoulder_vector = self._reference_xyz_m - SHOULDER_ORIGIN_M
        self._reference_radius_m = float(np.linalg.norm(shoulder_vector[:2]))
        self._reference_azimuth_rad = math.atan2(
            float(shoulder_vector[1]), float(shoulder_vector[0])
        )

        # Model conventions differ in the sign of base yaw. Probe FK once so
        # stick-right always changes the calibrated shoulder-pan target in the
        # same direction while the Cartesian target follows that pan.
        probe = measured.copy()
        pan_index = self.joint_names.index("shoulder_pan")
        probe[pan_index] += 1.0
        probe_xyz = np.asarray(
            self.kinematics.forward_kinematics(probe)[:3, 3], dtype=float
        )
        probe_vector = probe_xyz - SHOULDER_ORIGIN_M
        probe_azimuth = math.atan2(
            float(probe_vector[1]), float(probe_vector[0])
        )
        azimuth_delta = math.atan2(
            math.sin(probe_azimuth - self._reference_azimuth_rad),
            math.cos(probe_azimuth - self._reference_azimuth_rad),
        )
        self._pan_to_azimuth_sign = 1.0 if azimuth_delta >= 0.0 else -1.0

        self.shoulder_pan_target_deg = self._reference_pan_deg
        self.wrist_flex_target_deg = float(measured_positions["wrist_flex"])
        self.wrist_roll_target_deg = float(measured_positions["wrist_roll"])
        self._smoothed_axes.fill(0.0)

        if self.elbow_singularity_deg is not None:
            elbow_index = self.joint_names.index("elbow_flex")
            elbow_position = float(measured[elbow_index])
            self._elbow_branch_side = (
                1.0
                if elbow_position >= self.elbow_singularity_deg
                else -1.0
            )
            safe_elbow = (
                self.elbow_singularity_deg
                + self._elbow_branch_side
                * self.settings.extended_elbow_stop_deg
            )
            safe_elbow = float(
                np.clip(safe_elbow, *joint_limits_deg["elbow_flex"])
            )
            calibrated_lower, calibrated_upper = joint_limits_deg["elbow_flex"]
            self._active_elbow_ik_limits_deg = (
                (safe_elbow, calibrated_upper)
                if self._elbow_branch_side > 0.0
                else (calibrated_lower, safe_elbow)
            )
            self.kinematics.robot.set_joint_limits(
                "elbow_flex",
                *np.deg2rad(self._active_elbow_ik_limits_deg),
            )
            self.kinematics.solver.enable_joint_limits(True)
            safe_joints = measured.copy()
            safe_joints[elbow_index] = safe_elbow
            safe_wrist = np.asarray(
                self.kinematics.forward_kinematics(safe_joints)[:3, 3],
                dtype=float,
            )
            shoulder = np.asarray(
                self.kinematics.robot.get_T_world_frame(
                    SHOULDER_LIFT_FRAME
                )[:3, 3],
                dtype=float,
            )
            self._maximum_wrist_radius_m = float(
                np.linalg.norm(safe_wrist - shoulder)
            )

    def _update_translation_offset(self) -> None:
        assert self._reference_xyz_m is not None
        assert self._reference_pan_deg is not None
        assert self._reference_radius_m is not None
        assert self._reference_azimuth_rad is not None
        assert self.shoulder_pan_target_deg is not None

        radius = max(1e-4, self._reference_radius_m + self._planar_offset_m)
        pan_delta_rad = math.radians(
            self.shoulder_pan_target_deg - self._reference_pan_deg
        )
        azimuth = (
            self._reference_azimuth_rad
            + self._pan_to_azimuth_sign * pan_delta_rad
        )
        desired_xyz = np.asarray(
            [
                SHOULDER_ORIGIN_M[0] + radius * math.cos(azimuth),
                SHOULDER_ORIGIN_M[1] + radius * math.sin(azimuth),
                self._reference_xyz_m[2] + self._height_offset_m,
            ],
            dtype=float,
        )
        self.translation_offset_m = desired_xyz - self._reference_xyz_m

    def _project_to_reachable_workspace(self, measured: np.ndarray) -> bool:
        """Project an unreachable wrist target onto the safe reach sphere."""

        if self._maximum_wrist_radius_m is None:
            return False
        assert self._reference_xyz_m is not None
        assert self._reference_radius_m is not None
        assert self.shoulder_pan_target_deg is not None

        workspace_joints = measured.copy()
        workspace_joints[self.joint_names.index("shoulder_pan")] = (
            self.shoulder_pan_target_deg
        )
        self.kinematics.forward_kinematics(workspace_joints)
        shoulder = np.asarray(
            self.kinematics.robot.get_T_world_frame(
                SHOULDER_LIFT_FRAME
            )[:3, 3],
            dtype=float,
        )
        desired_xyz = self._reference_xyz_m + self.translation_offset_m
        shoulder_to_target = desired_xyz - shoulder
        requested_radius = float(np.linalg.norm(shoulder_to_target))
        if requested_radius <= self._maximum_wrist_radius_m + 1e-9:
            return False

        projected_xyz = (
            shoulder
            + shoulder_to_target
            * (self._maximum_wrist_radius_m / requested_radius)
        )
        projected_pan_radius = float(
            np.linalg.norm((projected_xyz - SHOULDER_ORIGIN_M)[:2])
        )
        self._planar_offset_m = projected_pan_radius - self._reference_radius_m
        self._height_offset_m = float(
            projected_xyz[2] - self._reference_xyz_m[2]
        )
        self.translation_offset_m = projected_xyz - self._reference_xyz_m
        return True

    def update(
        self,
        sample: GamepadSample,
        *,
        measured_positions: dict[str, float],
        joint_limits_deg: dict[str, tuple[float, float]],
    ) -> dict[str, float | bool]:
        now = float(sample.timestamp_s)
        self._cartesian_state_before_update = None
        gap = None if self._last_timestamp_s is None else now - self._last_timestamp_s
        initializing = self._reference_xyz_m is None
        stream_gap = gap is not None and gap > self.settings.reset_gap_s

        measured = np.asarray(
            [measured_positions[name] for name in self.joint_names], dtype=float
        )
        if initializing:
            self._latch_reference(
                measured,
                measured_positions,
                joint_limits_deg,
            )

        raw_shaped = np.asarray(
            [
                # This controller reports stick-forward as positive.
                shape_axis(
                    sample.left_y,
                    deadzone=self.settings.deadzone,
                    expo=self.settings.expo,
                ),
                shape_axis(
                    sample.left_x,
                    deadzone=self.settings.deadzone,
                    expo=self.settings.expo,
                ),
                float(np.clip(sample.dpad_vertical, -1, 1)),
                # Positive wrist flex lowers the gripper tip in the SO-101
                # model, so pushing this controller forward commands positive.
                shape_axis(
                    sample.right_y,
                    deadzone=self.settings.deadzone,
                    expo=self.settings.expo,
                ),
                shape_axis(
                    sample.right_x,
                    deadzone=self.settings.deadzone,
                    expo=self.settings.expo,
                ),
            ],
            dtype=float,
        )

        dt = 0.0
        workspace_clamped = False
        extension_clamped = False
        desired_absolute_pan_deg: float | None = None
        absolute_pan_tracking_active = False
        joint_limit_clamped: list[str] = []
        if (
            not initializing
            and gap is not None
            and gap >= 0.0
            and not stream_gap
        ):
            dt = float(
                np.clip(gap, self.settings.minimum_dt_s, self.settings.maximum_dt_s)
            )
            maximum_axis_change = self.settings.axis_slew_rate_per_s * dt
            self._smoothed_axes += np.clip(
                raw_shaped - self._smoothed_axes,
                -maximum_axis_change,
                maximum_axis_change,
            )
            requested_planar_offset = (
                self._planar_offset_m
                + self._smoothed_axes[0] * self.settings.planar_velocity_m_s * dt
            )
            requested_height_offset = (
                self._height_offset_m
                + self._smoothed_axes[2] * self.settings.height_velocity_m_s * dt
            )
            clipped_planar_offset = float(
                np.clip(
                    requested_planar_offset,
                    *self.settings.planar_offset_limits_m,
                )
            )
            clipped_height_offset = float(
                np.clip(
                    requested_height_offset,
                    *self.settings.height_offset_limits_m,
                )
            )
            self._cartesian_state_before_update = (
                self._planar_offset_m,
                self._height_offset_m,
            )
            workspace_clamped = not (
                math.isclose(requested_planar_offset, clipped_planar_offset)
                and math.isclose(requested_height_offset, clipped_height_offset)
            )

            assert self.shoulder_pan_target_deg is not None
            assert self.wrist_flex_target_deg is not None
            assert self.wrist_roll_target_deg is not None
            if self.settings.pan_control_mode == "absolute":
                pan_lower, pan_upper = joint_limits_deg["shoulder_pan"]
                pan_midpoint = 0.5 * (pan_lower + pan_upper)
                pan_half_span = 0.5 * (pan_upper - pan_lower)
                desired_absolute_pan_deg = (
                    pan_midpoint + self._smoothed_axes[1] * pan_half_span
                )
                absolute_pan_tracking_active = not math.isclose(
                    desired_absolute_pan_deg,
                    self.shoulder_pan_target_deg,
                    abs_tol=1e-9,
                )
                maximum_pan_step = (
                    self.settings.shoulder_pan_velocity_deg_s * dt
                )
                requested_pan = self.shoulder_pan_target_deg + float(
                    np.clip(
                        desired_absolute_pan_deg - self.shoulder_pan_target_deg,
                        -maximum_pan_step,
                        maximum_pan_step,
                    )
                )
            else:
                requested_pan = (
                    self.shoulder_pan_target_deg
                    + self._smoothed_axes[1]
                    * self.settings.shoulder_pan_velocity_deg_s
                    * dt
                )
            requested_direct_targets = {
                "shoulder_pan": requested_pan,
                "wrist_flex": (
                    self.wrist_flex_target_deg
                    + self._smoothed_axes[3]
                    * self.settings.wrist_flex_velocity_deg_s
                    * dt
                ),
                "wrist_roll": (
                    self.wrist_roll_target_deg
                    + self._smoothed_axes[4]
                    * self.settings.wrist_roll_velocity_deg_s
                    * dt
                ),
            }
            clipped_direct_targets = {
                name: float(np.clip(value, *joint_limits_deg[name]))
                for name, value in requested_direct_targets.items()
            }
            joint_limit_clamped = [
                name
                for name in requested_direct_targets
                if not math.isclose(
                    requested_direct_targets[name], clipped_direct_targets[name]
                )
            ]

            self._planar_offset_m = clipped_planar_offset
            self._height_offset_m = clipped_height_offset
            self.shoulder_pan_target_deg = clipped_direct_targets["shoulder_pan"]
            self.wrist_flex_target_deg = clipped_direct_targets["wrist_flex"]
            self.wrist_roll_target_deg = clipped_direct_targets["wrist_roll"]
            self._update_translation_offset()
            extension_clamped = self._project_to_reachable_workspace(measured)
            workspace_clamped = workspace_clamped or extension_clamped
        elif stream_gap:
            self._smoothed_axes.fill(0.0)

        assert self._reference_xyz_m is not None

        self._last_timestamp_s = now
        self.latest = {
            "controller": sample.controller_name,
            "raw_axes": {
                "left_x": sample.left_x,
                "left_y": sample.left_y,
                "right_x": sample.right_x,
                "right_y": sample.right_y,
                "left_trigger": sample.left_trigger,
                "right_trigger": sample.right_trigger,
                "dpad_vertical": sample.dpad_vertical,
            },
            "shaped_axes": {
                "planar_reach": float(self._smoothed_axes[0]),
                "shoulder_pan": float(self._smoothed_axes[1]),
                "height": float(self._smoothed_axes[2]),
                "wrist_flex": float(self._smoothed_axes[3]),
                "wrist_roll": float(self._smoothed_axes[4]),
            },
            "raw_shaped_axes": {
                "planar_reach": float(raw_shaped[0]),
                "shoulder_pan": float(raw_shaped[1]),
                "height": float(raw_shaped[2]),
                "wrist_flex": float(raw_shaped[3]),
                "wrist_roll": float(raw_shaped[4]),
            },
            "translation_offset_m": self.translation_offset_m.tolist(),
            "x_offset_m": float(self.translation_offset_m[0]),
            "y_offset_m": float(self.translation_offset_m[1]),
            "z_offset_m": float(self.translation_offset_m[2]),
            "planar_offset_m": self._planar_offset_m,
            "height_offset_m": self._height_offset_m,
            "workspace_clamped": workspace_clamped,
            "extension_clamped": extension_clamped,
            "workspace_projected": extension_clamped,
            "elbow_singularity_deg": self.elbow_singularity_deg,
            "maximum_wrist_radius_m": self._maximum_wrist_radius_m,
            "active_elbow_ik_limits_deg": self._active_elbow_ik_limits_deg,
            "joint_limit_clamped": joint_limit_clamped,
            "arm_input_active": bool(
                np.any(np.abs(self._smoothed_axes) > 1e-9)
                or absolute_pan_tracking_active
            ),
            "direct_joint_targets_deg": self.direct_joint_targets,
            "desired_absolute_pan_deg": desired_absolute_pan_deg,
            "gripper_direction": sample.gripper_direction,
            "episode_event": sample.episode_event,
            "dt_s": dt,
            "stream_gap": stream_gap,
            "settings": asdict(self.settings),
        }
        return {
            "enabled": True,
            "target_x": float(self.translation_offset_m[0]),
            "target_y": float(self.translation_offset_m[1]),
            "target_z": float(self.translation_offset_m[2]),
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper_vel": float(sample.gripper_direction),
        }


def diagnose_controller(controller_index: int = 0, duration_s: float = 8.0) -> None:
    """Print live raw state without importing or opening any robot class."""

    gamepad = PygameGamepad(controller_index)
    gamepad.connect()
    first = gamepad.read()
    print(
        f"Controller: {gamepad.name} "
        f"({len(first.raw_axes)} axes, {len(first.raw_buttons)} buttons)"
    )
    print("Move the sticks and press buttons. This diagnostic never opens the servo bus.")
    print(
        "SDL mapping: left stick=axes 0/1, right stick=axes 2/3, "
        "LT/RT=axes 4/5, B=base"
    )
    started = time.monotonic()
    try:
        while time.monotonic() - started < duration_s:
            sample = gamepad.read()
            print(
                f"axes={[round(value, 3) for value in sample.raw_axes]} "
                f"buttons={[index for index, value in enumerate(sample.raw_buttons) if value]} "
                f"LT={sample.left_trigger:+.3f}",
                f"RT={sample.right_trigger:+.3f}",
                f"gripper={sample.gripper_direction}",
                flush=True,
            )
            time.sleep(0.1)
    finally:
        gamepad.disconnect()
