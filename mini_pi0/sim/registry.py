from __future__ import annotations

from typing import Any

from mini_pi0.config.schema import RootConfig
from mini_pi0.sim.base import SimulatorAdapter
from mini_pi0.sim.isaaclab_adapter import IsaacLabAdapter
from mini_pi0.sim.maniskill3_adapter import ManiSkill3Adapter
from mini_pi0.sim.robosuite_adapter import RobosuiteAdapter

_SIM_REGISTRY = {
    "robosuite": RobosuiteAdapter,
    "maniskill3": ManiSkill3Adapter,
    "isaaclab": IsaacLabAdapter,
}


def list_backends() -> list[str]:
    """List simulator backend keys supported by the adapter registry.

    Returns:
        Sorted backend names.
    """

    return sorted(_SIM_REGISTRY.keys())


def make_sim_adapter(cfg: RootConfig) -> SimulatorAdapter:
    """Instantiate simulator adapter from config backend key.

    Args:
        cfg: Root configuration with ``simulator.backend`` populated.

    Returns:
        Backend-specific adapter instance.

    Raises:
        ValueError: If backend key is unknown.
    """

    key = str(cfg.simulator.backend).strip().lower()
    if key not in _SIM_REGISTRY:
        raise ValueError(f"Unknown simulator backend '{cfg.simulator.backend}'. Options: {list_backends()}")
    return _SIM_REGISTRY[key](cfg)


def backend_status() -> dict[str, dict[str, Any]]:
    """Return lightweight backend readiness diagnostics for CLI reporting.

    Returns:
        Mapping from backend name to readiness/status metadata.
    """

    out: dict[str, dict[str, Any]] = {}
    for name in list_backends():
        ok = True
        msg = "available"
        if name == "maniskill3":
            ok = False
            msg = "scaffolded only"
        elif name == "robosuite":
            try:
                import robosuite  # noqa: F401
            except Exception as e:
                ok = False
                msg = f"missing dependency: {type(e).__name__}"
        elif name == "isaaclab":
            ok = False
            msg = "scaffolded only"
        out[name] = {"ready": ok, "status": msg}
    return out
