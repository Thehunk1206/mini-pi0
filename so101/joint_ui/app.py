from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .controller import RobotController


DEFAULT_CALIBRATION = Path(
    os.environ.get(
        "SO101_CALIBRATION_FILE",
        "~/.cache/huggingface/lerobot/calibration/robots/so_follower/handy_bot.json",
    )
).expanduser()
DEFAULT_SERIAL_PORT = os.environ.get("SO101_SERIAL_PORT", "/dev/cu.usbmodem5B610338651")
DEFAULT_BASE_POSITION = Path(
    os.environ.get(
        "SO101_BASE_POSITION_FILE",
        f"~/.cache/huggingface/lerobot/base_positions/robots/so_follower/{DEFAULT_CALIBRATION.stem}.json",
    )
).expanduser()


class ConnectRequest(BaseModel):
    serial_port: str | None = None


class TargetRequest(BaseModel):
    joint: str
    value: float


class TorqueRequest(BaseModel):
    enabled: bool


def create_app(controller: RobotController) -> FastAPI:
    static_dir = Path(__file__).parent / "static"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        controller.disconnect()

    app = FastAPI(
        title="SO-101 Joint Control",
        description="Local calibrated joint dashboard for LeRobot SO-101",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.middleware("http")
    async def localhost_only(request, call_next):
        client_host = request.client.host if request.client else ""
        if client_host not in {"127.0.0.1", "::1", "testclient"}:
            return JSONResponse(
                status_code=403,
                content={"detail": "SO-101 control is restricted to localhost"},
            )
        return await call_next(request)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/state")
    def state() -> dict:
        return controller.snapshot()

    @app.post("/api/connect")
    def connect(request: ConnectRequest) -> dict:
        try:
            return controller.connect(request.serial_port)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/target")
    def target(request: TargetRequest) -> dict:
        try:
            return controller.set_target(request.joint, request.value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/torque")
    def torque(request: TorqueRequest) -> dict:
        try:
            return controller.set_torque(request.enabled)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/base-position/capture")
    def capture_base_position() -> dict:
        try:
            return controller.capture_base_position()
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/base-position/return")
    def return_to_base() -> dict:
        try:
            return controller.return_to_base()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/emergency-stop")
    def emergency_stop() -> dict:
        try:
            return controller.emergency_stop()
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/disconnect")
    def disconnect() -> dict:
        try:
            return controller.disconnect()
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


default_controller = RobotController(
    DEFAULT_CALIBRATION,
    DEFAULT_SERIAL_PORT,
    base_position_file=DEFAULT_BASE_POSITION,
)
app = create_app(default_controller)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local SO-101 joint-control dashboard")
    parser.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--base-position", type=Path, default=DEFAULT_BASE_POSITION)
    parser.add_argument("--web-port", type=int, default=8000)
    args = parser.parse_args()

    controller = RobotController(
        args.calibration,
        args.serial_port,
        base_position_file=args.base_position,
    )
    uvicorn.run(create_app(controller), host="127.0.0.1", port=args.web_port, log_level="info")


if __name__ == "__main__":
    main()
