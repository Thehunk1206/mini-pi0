from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any, Callable

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from mini_pi0.config.schema import RootConfig
from mini_pi0.dataset.obs_processor import ObsProcessor
from mini_pi0.sim.base import SimulatorAdapter


def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert rendered frame to ``uint8`` RGB-like array.

    Args:
        frame: Input frame in arbitrary numeric dtype/range.

    Returns:
        Frame converted/clipped to ``uint8``.
    """

    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
        return arr.astype(np.uint8)
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _reshape_action(action: np.ndarray, target_dim: int) -> np.ndarray:
    """Resize action vector to match simulator action dimension.

    If action is longer it is truncated; if shorter it is zero-padded.

    Args:
        action: Input action vector.
        target_dim: Required action dimension.

    Returns:
        Reshaped action vector of length ``target_dim``.
    """

    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] == target_dim:
        return a
    if a.shape[0] > target_dim:
        return a[:target_dim]
    out = np.zeros((target_dim,), dtype=np.float32)
    out[: a.shape[0]] = a
    return out


def _resolve_action_scale(scale: list[float] | None, target_dim: int) -> np.ndarray | None:
    """Resolve configured per-dimension action scale vector."""

    if scale is None:
        return None
    arr = np.asarray(scale, dtype=np.float32).reshape(-1)
    if arr.shape[0] != target_dim:
        return None
    return arr


def _classify_failure_reason(
    *,
    success: bool,
    max_step_reward: float,
    steps: int,
    max_steps: int | None,
    reward_threshold: float,
) -> str:
    """Classify rollout outcome into coarse reason buckets."""

    if success:
        return "success"
    if max_step_reward < float(reward_threshold):
        return "no_progress"
    if max_steps is not None and steps >= int(max_steps):
        return "timeout_after_progress"
    if max_step_reward >= 0.99:
        return "drop_or_unstable"
    return "failure"


def _bootstrap_ci_95(values: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    """Estimate 95% bootstrap confidence interval for sample mean.

    Args:
        values: Sample values.
        n_boot: Number of bootstrap resamples.

    Returns:
        ``(low, high)`` percentile bounds for 95% CI.
    """

    if values.size == 0:
        return 0.0, 0.0
    boot = [np.random.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def _format_duration(seconds: float) -> str:
    """Format seconds into human-friendly ``HH:MM:SS`` / ``MM:SS`` string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Compact wall-time string.
    """

    total = int(max(0, round(float(seconds))))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def evaluate(
    model: torch.nn.Module,
    processor: ObsProcessor,
    cfg: RootConfig,
    make_adapter: Callable[[int], SimulatorAdapter],
    collect_grid: bool = False,
) -> dict[str, Any] | tuple[dict[str, np.ndarray], dict[str, list[list[np.ndarray]]]]:
    """Run batched episodic evaluation and collect metrics.

    Args:
        model: Action model with ``sample`` method.
        processor: Observation/action normalization helper.
        cfg: Resolved root config.
        make_adapter: Factory that returns a seeded simulator adapter.
        collect_grid: Whether to collect rollout frames for success/failure grids.

    Returns:
        Either:
        - metrics dictionary of numpy arrays, or
        - ``(metrics, grid_rollouts)`` when ``collect_grid=True``.
    """

    model.eval()
    metrics: dict[str, list[float]] = defaultdict(list)
    n_episodes = int(cfg.eval.n_episodes)
    verbose = bool(cfg.eval.verbose)
    log_every = max(1, int(cfg.eval.log_every_episodes))
    success_count = 0
    start_time = time.perf_counter()
    smooth_alpha = float(max(0.0, min(1.0, getattr(cfg.eval, "action_smoothing_alpha", 0.0))))

    if verbose:
        print(
            "[eval] Starting evaluation: "
            f"episodes={n_episodes}, execute_steps={int(cfg.eval.execute_steps)}, "
            f"n_flow_steps={int(cfg.eval.n_flow_steps)}, max_steps={cfg.eval.max_steps}, "
            f"smoothing_alpha={smooth_alpha:.2f}",
            flush=True,
        )

    success_rollouts: list[list[np.ndarray]] = []
    failure_rollouts: list[list[np.ndarray]] = []
    grid_slots = int(max(1, cfg.eval.grid_size * cfg.eval.grid_size))

    for ep in range(n_episodes):
        adapter = make_adapter(ep)
        obs = adapter.reset(seed=ep)

        episode_rng = np.random.default_rng(ep)
        cube_xy = tuple(cfg.eval.cube_xy) if cfg.eval.cube_xy is not None else None
        cube_xy_range = tuple(cfg.eval.cube_xy_range) if cfg.eval.cube_xy_range is not None else None
        adapter.set_object_pose(
            object_name=cfg.task.object_name,
            xy=cube_xy,
            xy_range=cube_xy_range,
            z=cfg.eval.cube_z,
            yaw_deg=cfg.eval.cube_yaw_deg,
            rng=episode_rng,
        )

        action_buffer: list[np.ndarray] = []
        reward_sum = 0.0
        steps = 0
        success = False
        t_infer = 0.0
        chunks = 0

        frames: list[np.ndarray] | None = None
        if collect_grid:
            need_success = len(success_rollouts) < grid_slots
            need_failure = len(failure_rollouts) < grid_slots
            if need_success or need_failure:
                frames = []

        lo, hi = adapter.action_spec()
        action_dim = int(np.asarray(lo).reshape(-1).shape[0])
        action_scale = _resolve_action_scale(cfg.eval.action_scale, action_dim)
        prev_action: np.ndarray | None = None
        clip_count = 0
        action_count = 0
        action_abs_sum = 0.0
        action_abs_max = 0.0
        max_step_reward = float("-inf")

        while True:
            if not action_buffer:
                img, prop = processor.obs_to_tensors(obs)
                t0 = time.perf_counter()
                with torch.no_grad():
                    chunk = model.sample(img, prop, n_steps=int(cfg.eval.n_flow_steps)).squeeze(0)
                t_infer += time.perf_counter() - t0
                chunks += 1

                chunk = processor.denormalize(chunk).detach().cpu().numpy()
                proposed = []
                for a in chunk[: int(cfg.eval.execute_steps)]:
                    raw = _reshape_action(a, target_dim=action_dim)
                    if action_scale is not None:
                        raw = raw * action_scale
                    if prev_action is not None and smooth_alpha > 0.0:
                        raw = (1.0 - smooth_alpha) * raw + smooth_alpha * prev_action
                    clipped = np.clip(raw, lo, hi).astype(np.float32)
                    if np.any(np.abs(raw - clipped) > 1e-6):
                        clip_count += 1
                    proposed.append(clipped)
                action_buffer = proposed

            action = action_buffer.pop(0)
            prev_action = action.copy()
            action_count += 1
            action_abs = np.abs(action)
            action_abs_sum += float(action_abs.mean())
            action_abs_max = max(action_abs_max, float(action_abs.max()))

            step = adapter.step(action)
            obs = step.obs
            reward_sum += float(step.reward)
            max_step_reward = max(max_step_reward, float(step.reward))
            steps += 1

            if frames is not None:
                frame = adapter.render(
                    camera=cfg.simulator.camera_names[0],
                    width=int(cfg.eval.grid_width),
                    height=int(cfg.eval.grid_height),
                )
                frame = _ensure_uint8(frame)
                if adapter.backend_name == "robosuite":
                    frame = frame[::-1]
                frames.append(frame)

            step_success = adapter.check_success(info=step.info, obs=step.obs)
            if step_success:
                success = True

            done = bool(step.done or step_success)
            if cfg.eval.max_steps is not None and steps >= int(cfg.eval.max_steps):
                done = True
            if done:
                break

        if frames is not None:
            if success and len(success_rollouts) < grid_slots:
                success_rollouts.append(frames)
            elif (not success) and len(failure_rollouts) < grid_slots:
                failure_rollouts.append(frames)

        reason = _classify_failure_reason(
            success=success,
            max_step_reward=max_step_reward,
            steps=steps,
            max_steps=cfg.eval.max_steps,
            reward_threshold=float(getattr(cfg.eval, "failure_reward_threshold", 0.2)),
        )

        metrics["success"].append(float(success))
        metrics["episode_length"].append(float(steps))
        metrics["total_reward"].append(float(reward_sum))
        metrics["infer_ms"].append(float(1000.0 * t_infer / max(1, chunks)))
        metrics["max_step_reward"].append(float(max_step_reward))
        metrics["action_clip_fraction"].append(float(clip_count / max(1, action_count)))
        metrics["action_abs_mean"].append(float(action_abs_sum / max(1, action_count)))
        metrics["action_abs_max"].append(float(action_abs_max))
        metrics["failure_reason"].append(reason)
        adapter.close()
        if success:
            success_count += 1

        if verbose and ((ep + 1) % log_every == 0 or (ep + 1) == n_episodes):
            elapsed = time.perf_counter() - start_time
            eps_done = ep + 1
            rate_eps = eps_done / max(1e-9, elapsed)
            eta = (n_episodes - eps_done) / max(1e-9, rate_eps)
            running_sr = 100.0 * success_count / float(eps_done)
            status = "SUCCESS" if success else "FAILURE"
            print(
                f"[eval] episode {eps_done}/{n_episodes} | {status} "
                f"| steps={steps} | reward={reward_sum:.2f} "
                f"| infer={1000.0 * t_infer / max(1, chunks):.1f} ms/chunk "
                f"| reason={reason} | clip={100.0 * clip_count / max(1, action_count):.1f}% "
                f"| running_success={running_sr:.1f}% ({success_count}/{eps_done}) "
                f"| elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)}",
                flush=True,
            )

    if verbose:
        fail_count = n_episodes - success_count
        print(
            f"[eval] Completed: success={success_count}, failure={fail_count}, "
            f"success_rate={100.0 * success_count / max(1, n_episodes):.1f}%",
            flush=True,
        )

    out: dict[str, np.ndarray] = {}
    for k, v in metrics.items():
        if k == "failure_reason":
            out[k] = np.asarray(v, dtype=object)
        else:
            out[k] = np.asarray(v, dtype=np.float32)
    if collect_grid:
        return out, {"success": success_rollouts, "failure": failure_rollouts}
    return out


def save_rollout_grid(rollouts: list[list[np.ndarray]], path: str, grid_size: int = 3, fps: int = 20) -> None:
    """Save multiple rollout clips into an ``N x N`` tiled video.

    Args:
        rollouts: List of rollout clips; each clip is a list of frames.
        path: Output MP4 path.
        grid_size: Grid dimension ``N``.
        fps: Video frame rate.
    """

    grid_size = int(max(1, grid_size))
    n_cells = grid_size * grid_size
    selected = list(rollouts[:n_cells])
    if not selected:
        print(f"No rollouts available for grid video: {path}")
        return

    frame_shape = None
    for clip in selected:
        if clip:
            frame_shape = clip[0].shape
            break
    if frame_shape is None:
        print(f"No frames available for grid video: {path}")
        return

    h, w, c = frame_shape
    blank = np.zeros((h, w, c), dtype=np.uint8)

    while len(selected) < n_cells:
        selected.append([])

    max_t = max((len(clip) for clip in selected), default=0)
    if max_t <= 0:
        print(f"Rollouts are empty for grid video: {path}")
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for t in range(max_t):
            canvas = np.zeros((grid_size * h, grid_size * w, c), dtype=np.uint8)
            for idx, clip in enumerate(selected):
                r = idx // grid_size
                col = idx % grid_size
                frame = clip[min(t, len(clip) - 1)] if clip else blank
                canvas[r * h : (r + 1) * h, col * w : (col + 1) * w] = frame
            writer.append_data(canvas)

    print(f"Saved grid video ({grid_size}x{grid_size}) -> {path}")


def record_episode(
    model: torch.nn.Module,
    processor: ObsProcessor,
    cfg: RootConfig,
    make_adapter: Callable[[int], SimulatorAdapter],
    path: str,
    seed: int = 0,
) -> None:
    """Record a single rollout episode to MP4 for qualitative inspection.

    Args:
        model: Action model.
        processor: Observation/action normalization helper.
        cfg: Resolved root configuration.
        make_adapter: Factory returning simulator adapters.
        path: Output video path.
        seed: Episode seed.
    """

    adapter = make_adapter(seed)
    obs = adapter.reset(seed=seed)
    frames: list[np.ndarray] = []
    action_buffer: list[np.ndarray] = []

    lo, hi = adapter.action_spec()
    action_dim = int(np.asarray(lo).reshape(-1).shape[0])
    action_scale = _resolve_action_scale(cfg.eval.action_scale, action_dim)
    smooth_alpha = float(max(0.0, min(1.0, getattr(cfg.eval, "action_smoothing_alpha", 0.0))))
    prev_action: np.ndarray | None = None

    step_budget = int(cfg.eval.max_steps) if cfg.eval.max_steps is not None else int(cfg.simulator.horizon)

    for _ in range(step_budget):
        if not action_buffer:
            img, prop = processor.obs_to_tensors(obs)
            with torch.no_grad():
                chunk = model.sample(img, prop, n_steps=int(cfg.eval.n_flow_steps)).squeeze(0)
            chunk = processor.denormalize(chunk).detach().cpu().numpy()
            proposed = []
            for a in chunk[: int(cfg.eval.execute_steps)]:
                x = _reshape_action(a, target_dim=action_dim)
                if action_scale is not None:
                    x = x * action_scale
                if prev_action is not None and smooth_alpha > 0.0:
                    x = (1.0 - smooth_alpha) * x + smooth_alpha * prev_action
                x = np.clip(x, lo, hi).astype(np.float32)
                proposed.append(x)
            action_buffer = proposed

        action = action_buffer.pop(0)
        prev_action = action.copy()
        step = adapter.step(action)
        obs = step.obs

        frame = adapter.render(camera=cfg.simulator.camera_names[0], width=512, height=512)
        frame = _ensure_uint8(frame)
        if adapter.backend_name == "robosuite":
            frame = frame[::-1]
        frames.append(frame)

        success = adapter.check_success(info=step.info, obs=step.obs)
        if step.done or success:
            break

    adapter.close()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with imageio.get_writer(path, fps=20) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"Saved {len(frames)} frames -> {path}")


def report(results: dict[str, np.ndarray], plot_path: str = "eval_metrics.png") -> dict[str, float]:
    """Summarize evaluation arrays and write metrics visualization.

    Args:
        results: Metrics arrays output by ``evaluate``.
        plot_path: Destination path for PNG summary figure.

    Returns:
        Scalar summary dictionary used by CLI/reporting.
    """

    sr = np.asarray(results["success"], dtype=np.float32)
    el = np.asarray(results["episode_length"], dtype=np.float32)
    rw = np.asarray(results["total_reward"], dtype=np.float32)
    inf = np.asarray(results["infer_ms"], dtype=np.float32)
    mxr = np.asarray(results.get("max_step_reward", np.array([])), dtype=np.float32)
    clipf = np.asarray(results.get("action_clip_fraction", np.array([])), dtype=np.float32)
    aabs = np.asarray(results.get("action_abs_mean", np.array([])), dtype=np.float32)
    reasons = np.asarray(results.get("failure_reason", np.array([])), dtype=object)

    lo, hi = _bootstrap_ci_95(sr)
    reason_counts: dict[str, int] = {}
    if reasons.size > 0:
        for r in reasons.tolist():
            key = str(r)
            reason_counts[key] = reason_counts.get(key, 0) + 1

    summary = {
        "success_rate": float(sr.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "episode_len_mean": float(el.mean()),
        "episode_len_std": float(el.std()),
        "reward_mean": float(rw.mean()),
        "infer_ms_mean": float(inf.mean()),
        "max_step_reward_mean": float(mxr.mean()) if mxr.size > 0 else 0.0,
        "action_clip_fraction_mean": float(clipf.mean()) if clipf.size > 0 else 0.0,
        "action_abs_mean": float(aabs.mean()) if aabs.size > 0 else 0.0,
        "failure_reason_counts": reason_counts,
    }

    print(f"Success rate : {summary['success_rate'] * 100:.1f}%  [{summary['ci95_low'] * 100:.1f}%, {summary['ci95_high'] * 100:.1f}%] CI95")
    print(f"Episode len  : {summary['episode_len_mean']:.0f} ± {summary['episode_len_std']:.0f} steps")
    print(f"Reward       : {summary['reward_mean']:.2f}")
    print(f"Infer speed  : {summary['infer_ms_mean']:.1f} ms/chunk")
    print(f"Max step rwd : {summary['max_step_reward_mean']:.3f}")
    print(f"Action clip  : {summary['action_clip_fraction_mean'] * 100:.1f}%")
    if reason_counts:
        print(f"Failure mode : {reason_counts}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    sr_pct = sr * 100.0
    x = np.arange(len(sr_pct))
    axes[0].plot(x, sr_pct, marker="o", linestyle="-", alpha=0.35, label="raw")

    window = min(10, max(2, len(sr_pct) // 3)) if len(sr_pct) >= 2 else 1
    if window > 1 and len(sr_pct) >= window:
        smooth = np.convolve(sr_pct, np.ones(window) / window, mode="valid")
        smooth_x = np.arange(window - 1, len(sr_pct))
        axes[0].plot(smooth_x, smooth, linewidth=2.0, label=f"rolling({window})")
    else:
        axes[0].plot(x, sr_pct, marker="o", linewidth=2.0, label="mean")

    axes[0].set_title("Success rate by episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("%")
    axes[0].set_ylim(-2, 102)
    axes[0].legend()

    mask = sr.astype(bool)
    ep_idx = np.arange(len(el))
    axes[1].scatter(ep_idx[mask], el[mask], color="#1D9E75", alpha=0.9, label=f"success ({mask.sum()})")
    axes[1].scatter(ep_idx[~mask], el[~mask], color="#D85A30", alpha=0.9, label=f"failure ({(~mask).sum()})")
    if mask.any():
        axes[1].axhline(el[mask].mean(), color="#1D9E75", linestyle="--", alpha=0.6, linewidth=1.5)
    if (~mask).any():
        axes[1].axhline(el[~mask].mean(), color="#D85A30", linestyle="--", alpha=0.6, linewidth=1.5)
    axes[1].legend()
    axes[1].set_title("Episode length by episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")

    axes[2].hist(rw, bins=20, color="#3B82F6", alpha=0.8)
    axes[2].set_title("Reward distribution")

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return summary
