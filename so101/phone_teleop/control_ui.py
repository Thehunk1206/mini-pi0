"""Local desktop telemetry and URDF panel for SO-101 phone teleoperation."""

from __future__ import annotations

import copy
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


class RuntimeControlState:
    """Thread-safe bridge between the HTTP UI and the serial control loop."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._return_base_requested = False
        self._live: dict[str, Any] = {
            "connected": False,
            "phase": "startup",
            "phone_enabled": False,
            "loop_ms": None,
            "positions": {},
            "commands": {},
            "electrical": {},
            "cartesian": {},
            "control_mapping": {
                "orientation_enabled": False,
                "translation_gain": None,
                "max_ee_step_m": None,
                "gripper_speed_factor": None,
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

    def publish(self, **values: Any) -> None:
        with self._lock:
            self._live.update(copy.deepcopy(values))


def create_app(state: RuntimeControlState, ready_event: threading.Event) -> FastAPI:
    static_dir = Path(__file__).parent / "dashboard"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ready_event.set()
        yield

    app = FastAPI(
        title="SO-101 Phone Teleoperation",
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
        return await call_next(request)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        return state.snapshot()

    @app.post("/api/return-to-base")
    def return_to_base() -> dict[str, Any]:
        state.request_base_return()
        return state.snapshot()

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
            name="so101-phone-control-ui",
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
