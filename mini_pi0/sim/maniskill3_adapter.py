from __future__ import annotations

import importlib

from mini_pi0.config.schema import RootConfig
from mini_pi0.sim.base import SimulatorAdapter, StepOutput


class ManiSkill3Adapter(SimulatorAdapter):
    """Scaffold adapter for ManiSkill3 backend integration."""

    backend_name = "maniskill3"

    def __init__(self, cfg: RootConfig):
        """Create scaffold adapter and perform dependency availability check.

        Args:
            cfg: Root configuration object.
        """

        self.cfg = cfg
        self._has_module = importlib.util.find_spec("mani_skill") is not None

    def _raise(self):
        """Raise informative scaffold/runtime errors for unimplemented methods."""

        if not self._has_module:
            raise RuntimeError(
                "ManiSkill3 adapter is scaffolded. ManiSkill3 is not installed in this environment."
            )
        raise NotImplementedError(
            "ManiSkill3 adapter is currently scaffolded in this repository. "
            "Enable runtime implementation on a GPU-capable instance and wire task-specific mappings."
        )

    def reset(self, seed=None):
        """Scaffold reset method placeholder.

        Args:
            seed: Optional reset seed.
        """

        self._raise()

    def step(self, action) -> StepOutput:
        """Scaffold step method placeholder.

        Args:
            action: Action vector.
        """

        self._raise()

    def action_spec(self):
        """Scaffold action bounds method placeholder."""

        self._raise()

    def render(self, camera: str = "agentview", width: int = 512, height: int = 512):
        """Scaffold render method placeholder.

        Args:
            camera: Camera name.
            width: Frame width.
            height: Frame height.
        """

        self._raise()

    def check_success(self, info=None, obs=None) -> bool:
        """Scaffold success-check method placeholder.

        Args:
            info: Backend info dictionary.
            obs: Canonical observation dictionary.
        """

        self._raise()

    def close(self) -> None:
        """No-op close for scaffold adapter."""

        return None
