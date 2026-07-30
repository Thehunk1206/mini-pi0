from __future__ import annotations

from typing import Any

from mini_pi0.config.schema import RootConfig
from mini_pi0.sim.base import SimulatorAdapter

_SIM_REGISTRY = {
    "maniskill3": "mini_pi0.sim.maniskill3_adapter:ManiSkill3Adapter",
    "isaaclab": "mini_pi0.sim.isaaclab_adapter:IsaacLabAdapter",
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
    cls = _load_adapter_class(_SIM_REGISTRY[key])
    return cls(cfg)


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
            try:
                import mani_skill  # noqa: F401

                ok = True
                msg = "implemented"
            except Exception as e:
                ok = False
                msg = f"implemented, missing dependency: {type(e).__name__}"
        elif name == "isaaclab":
            try:
                from mini_pi0.sim.isaaclab_adapter import isaaclab_available

                ok = isaaclab_available()
                msg = "implemented" if ok else "implemented, missing dependency: isaaclab"
            except Exception as e:
                ok = False
                msg = f"implemented, status check failed: {type(e).__name__}"
        out[name] = {"ready": ok, "status": msg}
    return out


def _load_adapter_class(path: str) -> type[SimulatorAdapter]:
    """Load an adapter class from ``module:class`` path."""

    module_name, class_name = path.split(":", maxsplit=1)
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls
