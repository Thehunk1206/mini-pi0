"""Shared local desktop telemetry and URDF panel for SO-101 teleoperation."""

from __future__ import annotations

import copy
import threading
import webbrowser
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from so101.teleop.model_assets import DEFAULT_MODEL_CACHE, MODEL_FILENAME
from so101.teleop.profiles import PROFILE_NAMES
from so101.phone_teleop.filtering import (
    DEFAULT_PHONE_FILTER_SETTINGS,
    PHONE_FILTER_SETTING_BOUNDS,
    validated_phone_filter_settings,
)


class RuntimeControlState:
    """Thread-safe bridge between the HTTP UI and the serial control loop."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._return_base_requested = False
        self._settings_requested: dict[str, Any] | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=1800)
        self._history_sequence = 0
        self._live: dict[str, Any] = {
            "connected": False,
            "phase": "startup",
            "control_source": "phone",
            "phone_enabled": False,
            "loop_ms": None,
            "positions": {},
            "commands": {},
            "electrical": {},
            "cartesian": {},
            "control": {},
            "active_profile": "Safe",
            "filter_settings": dict(DEFAULT_PHONE_FILTER_SETTINGS),
            "calibration": {},
            "control_mapping": {
                "orientation_enabled": False,
                "translation_gain": None,
                "max_ee_step_m": None,
                "gripper_speed_factor": None,
                "translation_mapping": "LeRobot Android default",
            },
            "robot": {
                "name": "SO-101",
                "root_link": "base_link",
                "edges": [],
                "actual_links_m": {},
                "target_links_m": {},
            },
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                **copy.deepcopy(self._live),
                "return_base_pending": self._return_base_requested,
            }

    def request_base_return(self) -> None:
        with self._lock:
            self._return_base_requested = True

    def consume_base_return(self) -> bool:
        with self._lock:
            requested = self._return_base_requested
            self._return_base_requested = False
            return requested

    def request_settings(self, settings: dict[str, Any]) -> None:
        with self._lock:
            is_gamepad = self._live.get("control_source") == "gamepad"
            if is_gamepad:
                raise ValueError(
                    "Gamepad direct mode has no Ruckig motion profile or phone filter settings"
                )
            if bool(self._live.get("phone_enabled", False)):
                raise RuntimeError("Release Hold before changing motion settings")
            profile = settings.get("profile", self._live.get("active_profile", "Safe"))
            if profile not in PROFILE_NAMES:
                raise ValueError(f"Unknown motion profile: {profile}")
            filter_updates = {
                name: value
                for name, value in settings.items()
                if name in PHONE_FILTER_SETTING_BOUNDS
            }
            orientation_enabled = settings.get(
                "orientation_enabled",
                self._live.get("control_mapping", {}).get(
                    "orientation_enabled", False
                ),
            )
            if not isinstance(orientation_enabled, bool):
                raise ValueError("orientation_enabled must be true or false")
            unknown = set(settings) - {
                "profile",
                "orientation_enabled",
                *PHONE_FILTER_SETTING_BOUNDS,
            }
            if unknown:
                raise ValueError(f"Unsupported live setting(s): {', '.join(sorted(unknown))}")
            filter_settings = validated_phone_filter_settings(
                filter_updates,
                base=self._live.get(
                    "filter_settings", DEFAULT_PHONE_FILTER_SETTINGS
                ),
            )
            self._settings_requested = {
                "profile": profile,
                "filter_settings": filter_settings,
                "orientation_enabled": orientation_enabled,
            }

    def consume_settings(self) -> dict[str, Any] | None:
        with self._lock:
            requested = self._settings_requested
            self._settings_requested = None
            return copy.deepcopy(requested)

    def history(self, after_sequence: int | None = None) -> dict[str, Any]:
        """Return a full or incremental immutable snapshot of plot history.

        History entries are deep-copied when they enter the deque and are never
        mutated afterward.  Taking only a shallow list snapshot here keeps the
        HTTP thread from holding the control-loop lock while copying an
        ever-growing nested payload.
        """
        with self._lock:
            samples = list(self._history)
            latest_sequence = self._history_sequence

        oldest_sequence = samples[0]["sequence"] if samples else latest_sequence + 1
        reset = (
            after_sequence is not None
            and after_sequence < oldest_sequence - 1
        )
        if after_sequence is not None and not reset:
            samples = [
                sample
                for sample in samples
                if sample["sequence"] > after_sequence
            ]
        return {
            "samples": samples,
            "latest_sequence": latest_sequence,
            "reset": reset,
        }

    def publish(self, **values: Any) -> None:
        with self._lock:
            self._live.update(copy.deepcopy(values))
            if "positions" in values or "commands" in values:
                self._history_sequence += 1
                self._history.append(
                    {
                        "sequence": self._history_sequence,
                        **{
                            key: copy.deepcopy(self._live.get(key))
                            for key in (
                                "loop_ms", "positions", "commands", "electrical",
                                "cartesian", "control", "active_profile",
                            )
                        },
                    }
                )


def create_app(
    state: RuntimeControlState,
    ready_event: threading.Event,
    *,
    model_cache: Path = DEFAULT_MODEL_CACHE,
) -> FastAPI:
    static_dir = Path(__file__).parents[1] / "phone_teleop" / "dashboard"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ready_event.set()
        yield

    app = FastAPI(
        title="SO-101 Teleoperation",
        description="Local runtime telemetry and URDF visualization",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.middleware("http")
    async def localhost_only(request, call_next):
        client_host = request.client.host if request.client else ""
        if client_host not in {"127.0.0.1", "::1", "testclient"}:
            return JSONResponse(
                status_code=403,
                content={"detail": "SO-101 teleoperation controls are restricted to localhost"},
            )
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/meta")
    def get_meta() -> dict[str, Any]:
        return {
            "mode": "live",
            "model_url": f"/model/{MODEL_FILENAME}",
            "joint_names": [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper",
            ],
        }

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        return state.snapshot()

    @app.get("/api/history")
    def get_history(after_sequence: int | None = None) -> dict[str, Any]:
        return state.history(after_sequence)

    @app.post("/api/return-to-base")
    def return_to_base() -> dict[str, Any]:
        state.request_base_return()
        return state.snapshot()

    @app.put("/api/settings")
    @app.post("/api/settings")
    def update_settings(settings: dict[str, Any]) -> dict[str, Any]:
        try:
            state.request_settings(settings)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"accepted": True, "settings": settings}

    @app.get("/model/{asset_path:path}", include_in_schema=False)
    def model_asset(asset_path: str) -> FileResponse:
        root = Path(model_cache).resolve()
        requested = (root / asset_path).resolve()
        if not requested.is_relative_to(root) or not requested.is_file():
            raise HTTPException(status_code=404, detail="SO-101 model asset not found")
        media_type = "application/xml" if requested.suffix.lower() == ".urdf" else "model/stl"
        return FileResponse(requested, media_type=media_type)

    return app


class DesktopControlServer:
    """Run the localhost UI in a background thread without touching the servo bus."""

    def __init__(
        self,
        state: RuntimeControlState,
        *,
        host: str = "127.0.0.1",
        port: int = 8001,
    ) -> None:
        self.state = state
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}"
        self._ready = threading.Event()
        self._server = uvicorn.Server(
            uvicorn.Config(
                create_app(state, self._ready),
                host=host,
                port=port,
                log_level="warning",
            )
        )
        self._thread = threading.Thread(
            target=self._server.run,
            name="so101-teleop-control-ui",
            daemon=True,
        )

    def start(self, *, open_browser: bool = True) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            self._server.should_exit = True
            raise RuntimeError(f"Desktop control UI did not start at {self.url}")
        if open_browser:
            webbrowser.open(self.url)

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
