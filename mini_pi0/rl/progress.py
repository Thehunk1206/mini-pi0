"""Interactive terminal progress for ReinFlow rollout and PPO updates."""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, TextIO

from tqdm import tqdm

if TYPE_CHECKING:
    from mini_pi0.rl.flow_ppo import ReinFlowPPOProgress


class ReinFlowProgressDisplay:
    """Render one update's rollout and optimization progress in a TTY."""

    def __init__(
        self,
        *,
        update: int,
        total_updates: int,
        rollout_decisions: int,
        enabled: bool,
        stream: TextIO | None = None,
    ) -> None:
        """Create a rollout progress bar that remains silent in redirected logs."""

        self.update = int(update)
        self.total_updates = int(total_updates)
        self.stream = stream or sys.stderr
        self.enabled = bool(enabled) and self.stream.isatty()
        self._stage = "rollout"
        self._stage_completed = 0
        self._bar = self._make_bar("rollout", int(rollout_decisions), "cyan")
        self._started_at = time.perf_counter()

    def advance_rollout(
        self,
        *,
        primitive_steps: int,
        macro_reward: float,
        completed_episodes: int,
        success_rate: float | None,
    ) -> None:
        """Advance collection by one vectorized macro decision."""

        if self._bar is None:
            return
        elapsed = max(time.perf_counter() - self._started_at, 1e-9)
        success = "n/a" if success_rate is None else f"{success_rate:.1%}"
        self._bar.set_postfix(
            {
                "steps": str(int(primitive_steps)),
                "steps/s": f"{primitive_steps / elapsed:.1f}",
                "reward": f"{macro_reward:.3f}",
                "episodes": str(int(completed_episodes)),
                "success": success,
            },
            refresh=False,
        )
        self._bar.update(1)

    def finish_rollout(self) -> None:
        """Close the rollout bar before likelihood rebasing starts."""

        self._close_bar()

    def advance_ppo(self, event: ReinFlowPPOProgress) -> None:
        """Render one likelihood-rebase or optimizer minibatch event."""

        if not self.enabled:
            return
        if event.stage != self._stage:
            self._close_bar()
            colour = "yellow" if event.stage == "rebase" else "magenta"
            self._bar = self._make_bar(event.stage, event.total, colour)
            self._stage = event.stage
            self._stage_completed = 0
        if self._bar is None:
            return
        self._bar.set_postfix(self._ppo_postfix(event), refresh=False)
        increment = max(0, int(event.completed) - self._stage_completed)
        self._bar.update(increment)
        self._stage_completed = int(event.completed)

    def close(self) -> None:
        """Close the active progress bar."""

        self._close_bar()

    def _make_bar(self, stage: str, total: int, colour: str):
        """Create a consistently formatted stage bar."""

        if not self.enabled:
            return None
        return tqdm(
            total=max(1, int(total)),
            desc=f"[reinflow {self.update:04d}/{self.total_updates:04d}] {stage}",
            unit="batch" if stage != "rollout" else "decision",
            dynamic_ncols=True,
            mininterval=0.5,
            colour=colour,
            file=self.stream,
            leave=True,
        )

    @staticmethod
    def _ppo_postfix(event: ReinFlowPPOProgress) -> dict[str, str]:
        """Build concise live metrics for one PPO stage."""

        if event.stage == "rebase":
            return {"max correction": f"{event.log_prob_correction:.3g}"}
        policy = "off" if event.policy_loss is None else f"{event.policy_loss:.4f}"
        kl = "n/a" if event.approx_kl is None else f"{event.approx_kl:.3g}"
        return {
            "policy": policy,
            "value": f"{event.value_loss:.3f}",
            "kl": kl,
            "actor steps": str(event.actor_steps),
        }

    def _close_bar(self) -> None:
        """Close and clear the current tqdm instance."""

        if self._bar is not None:
            self._bar.close()
            self._bar = None


def terminal_color(text: str, code: str, *, stream: TextIO | None = None) -> str:
    """Apply ANSI color only when output is an interactive color terminal."""

    output = stream or sys.stdout
    if not output.isatty() or os.environ.get("NO_COLOR") is not None:
        return text
    return f"\033[{code}m{text}\033[0m"
