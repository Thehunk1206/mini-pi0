from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import torch
from numpy.lib.format import write_array

from mini_pi0.config.io import dump_config
from mini_pi0.config.schema import RootConfig
from mini_pi0.dataset.episodes import iter_lerobot_episode_images, load_episodes_from_config
from mini_pi0.utils.device import resolve_device
from mini_pi0.utils.runs import create_run_dir
from mini_pi0.vision.encoders import build_vision_extractor, images_to_tensor

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _chunked(items: list[np.ndarray], size: int):
    n = max(1, int(size))
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _pack_episode_npy_to_npz(episodes_dir: Path, out_npz: Path, n_episodes: int) -> None:
    """Pack per-episode ``.npy`` feature files into a single ``.npz`` archive.

    Args:
        episodes_dir: Directory containing ``ep_XXXXXX.npy`` files.
        out_npz: Destination archive path.
        n_episodes: Number of expected episodes to pack.
    """

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    if out_npz.exists():
        out_npz.unlink()

    with zipfile.ZipFile(out_npz, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for ep_idx in range(int(n_episodes)):
            key = f"ep_{ep_idx:06d}"
            ep_file = episodes_dir / f"{key}.npy"
            if not ep_file.exists():
                raise FileNotFoundError(f"Missing episodic feature file while packing NPZ: {ep_file}")
            arr = np.load(ep_file)
            with zf.open(f"{key}.npy", mode="w") as f:
                write_array(f, arr, allow_pickle=False)


def run_precompute_vision(cfg: RootConfig) -> dict[str, Any]:
    """Precompute per-timestep vision features and store them on disk.

    Args:
        cfg: Resolved repository configuration.

    Returns:
        Summary dictionary with output paths and feature metadata.
    """

    run_dir = create_run_dir(cfg.experiment.runs_root, f"{cfg.experiment.name}-vision")
    dump_config(run_dir / "config_resolved.yaml", cfg)

    fmt = str(getattr(cfg.data, "format", "robomimic_hdf5")).strip().lower()
    use_lerobot_stream = fmt in {"lerobot", "lerobot_hf", "hf"}
    episodes = None
    episode_frames_iter = None
    total_episodes_hint: int | None = None
    total_frames_hint: int | None = None

    if use_lerobot_stream:
        episode_frames_iter, meta = iter_lerobot_episode_images(
            repo_id=cfg.data.lerobot_repo_id or "",
            image_key=cfg.robot.image_key,
            episode_index_key=cfg.data.lerobot_episode_index_key,
            limit=cfg.data.n_demos,
            fallback_image_hw=tuple(cfg.data.fallback_image_hw),
            local_files_only=bool(cfg.data.lerobot_local_files_only),
            video_backend=cfg.data.lerobot_video_backend,
        )
        total_episodes_hint = int(cfg.data.n_demos) if cfg.data.n_demos is not None else meta.get("total_episodes")
        # When episode limit is set, full-dataset frame count becomes a misleading progress target.
        total_frames_hint = None if cfg.data.n_demos is not None else meta.get("total_frames")
    else:
        # Always source raw image observations for feature extraction.
        data_cfg = cfg.data
        prev_mode = data_cfg.observation_mode
        data_cfg.observation_mode = "image"
        episodes = load_episodes_from_config(cfg)
        data_cfg.observation_mode = prev_mode
        total_episodes_hint = len(episodes)
        total_frames_hint = int(sum(len(ep.obs) for ep in episodes))

    device = resolve_device(cfg.train.device)
    batch_size = max(1, int(cfg.vision.batch_size))
    print(
        "[vision] Precompute start | "
        f"backend={cfg.vision.backend} model={cfg.vision.model_name} "
        f"image_size={int(cfg.vision.image_size)} batch_size={batch_size} device={device}",
        flush=True,
    )
    print(
        "[vision] Dataset source | "
        f"format={cfg.data.format} n_demos={cfg.data.n_demos} image_key={cfg.robot.image_key}",
        flush=True,
    )
    if use_lerobot_stream:
        print("[vision] Loader mode | streaming (LeRobot episode-by-episode, low-memory)", flush=True)
    else:
        print("[vision] Loader mode | in-memory episodes", flush=True)

    extractor = build_vision_extractor(
        backend=cfg.vision.backend,
        model_name=cfg.vision.model_name,
        pretrained=bool(cfg.vision.pretrained),
        image_size=int(cfg.vision.image_size),
        hf_model_id=cfg.vision.hf_model_id,
        local_files_only=bool(cfg.vision.local_files_only),
        device=device,
    )

    image_key = cfg.robot.image_key
    zero_frames = 0
    total_frames = 0
    total_eps = 0
    feature_dim: int | None = None

    requested_out = Path(cfg.data.precomputed_features_path or cfg.vision.output_path)
    writes_npz = requested_out.suffix.lower() == ".npz"
    episodic_out_dir = (
        requested_out.with_suffix(requested_out.suffix + ".episodes")
        if writes_npz
        else requested_out
    )
    episodic_out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[vision] Loaded episodes={total_episodes_hint if total_episodes_hint is not None else '?'} "
        f"total_frames={total_frames_hint if total_frames_hint is not None else '?'}",
        flush=True,
    )

    if use_lerobot_stream:
        episode_iter = episode_frames_iter
    else:
        episode_iter = (
            (ep_idx, [np.asarray(obs[image_key], dtype=np.uint8) for obs in ep.obs]) for ep_idx, ep in enumerate(episodes)
        )

    if tqdm is not None:
        episode_iter = tqdm(
            episode_iter,
            total=total_episodes_hint,
            desc="Episodes",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        frame_bar = tqdm(
            total=total_frames_hint,
            desc="Frames",
            unit="frame",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
    else:
        frame_bar = None

    for ep_idx, frames in episode_iter:
        if not frames:
            if frame_bar is not None:
                frame_bar.update(0)
            continue
        total_eps += 1
        total_frames += len(frames)
        zero_frames += int(sum(int(np.max(f) == 0) for f in frames))

        feats = []
        for batch in _chunked(frames, batch_size):
            x = images_to_tensor(batch, device=device)
            with torch.no_grad():
                y = extractor(x)
            feats.append(y.detach().cpu().numpy().astype(np.float32))
            if frame_bar is not None:
                frame_bar.update(len(batch))
        arr = np.concatenate(feats, axis=0)
        key = f"ep_{ep_idx:06d}"
        np.save(episodic_out_dir / f"{key}.npy", arr)
        if feature_dim is None:
            feature_dim = int(arr.shape[-1])
        if tqdm is not None:
            episode_iter.set_postfix(feature_dim=int(arr.shape[-1]), cached_eps=total_eps)
        elif (ep_idx + 1) % 10 == 0:
            print(f"[vision] Processed {ep_idx + 1} episodes")

    if frame_bar is not None:
        frame_bar.close()

    if writes_npz:
        _pack_episode_npy_to_npz(episodic_out_dir, requested_out, total_eps)
        out_path = requested_out
    else:
        out_path = episodic_out_dir

    manifest = {
        "path": str(out_path),
        "episodic_path": str(episodic_out_dir),
        "storage": "npz+episodic_npy" if writes_npz else "episodic_npy",
        "feature_key": cfg.data.precomputed_feature_key,
        "n_episodes": int(total_eps),
        "feature_dim": int(feature_dim if feature_dim is not None else extractor.feature_dim),
        "encoder": {
            "backend": extractor.backend,
            "model_name": extractor.model_name,
            "image_size": int(extractor.image_size),
            "pretrained": bool(cfg.vision.pretrained),
            "hf_model_id": cfg.vision.hf_model_id,
        },
    }

    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with (artifacts_dir / "vision_features_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with (run_dir / "metrics" / "vision_precompute_summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if total_frames > 0 and (zero_frames / total_frames) > 0.9:
        print(
            "[warning] Most input frames are zeros. "
            "This usually means your dataset is low_dim without image observations.",
            flush=True,
        )

    print(f"[vision] Saved precomputed vision features: {out_path}", flush=True)
    print(
        f"[vision] Episodes: {manifest['n_episodes']} | feature_dim: {manifest['feature_dim']} "
        f"| encoded_frames={total_frames}",
        flush=True,
    )
    print(f"Run artifacts saved under: {run_dir}", flush=True)

    return {
        "run_dir": str(run_dir),
        "features_path": str(out_path),
        "manifest": manifest,
    }
