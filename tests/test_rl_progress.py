"""Tests for interactive ReinFlow terminal progress."""

from __future__ import annotations

from io import StringIO

from mini_pi0.rl.flow_ppo import ReinFlowPPOProgress
from mini_pi0.rl.progress import ReinFlowProgressDisplay


class _InteractiveBuffer(StringIO):
    """String buffer that behaves like an interactive terminal."""

    def isatty(self) -> bool:
        """Report TTY support so tqdm renders during the test."""

        return True


def test_progress_display_reports_rollout_rebase_and_optimize_stages() -> None:
    stream = _InteractiveBuffer()
    display = ReinFlowProgressDisplay(
        update=2,
        total_updates=10,
        rollout_decisions=1,
        enabled=True,
        stream=stream,
    )

    display.advance_rollout(
        primitive_steps=64,
        macro_reward=2.5,
        completed_episodes=2,
        success_rate=0.5,
    )
    display.finish_rollout()
    display.advance_ppo(
        ReinFlowPPOProgress(
            stage="rebase",
            completed=1,
            total=1,
            log_prob_correction=0.004,
        )
    )
    display.advance_ppo(
        ReinFlowPPOProgress(
            stage="optimize",
            completed=1,
            total=1,
            value_loss=3.0,
            policy_loss=-0.1,
            approx_kl=0.002,
            actor_steps=1,
        )
    )
    display.close()

    output = stream.getvalue()
    assert "[reinflow 0002/0010] rollout" in output
    assert "success=50.0%" in output
    assert "[reinflow 0002/0010] rebase" in output
    assert "[reinflow 0002/0010] optimize" in output


def test_progress_display_is_silent_for_noninteractive_stream() -> None:
    stream = StringIO()
    display = ReinFlowProgressDisplay(
        update=1,
        total_updates=1,
        rollout_decisions=1,
        enabled=True,
        stream=stream,
    )

    display.advance_rollout(
        primitive_steps=1,
        macro_reward=1.0,
        completed_episodes=0,
        success_rate=None,
    )
    display.close()

    assert stream.getvalue() == ""
