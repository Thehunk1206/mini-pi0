"""Checkpoint loading and observation preprocessing for SO-101 policies."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from mini_pi0.config.schema import ModelConfig
from mini_pi0.dataset.image_transforms import resize_image_tensor
from mini_pi0.models.registry import count_params, make_model
from mini_pi0.utils.device import resolve_device
from mini_pi0.utils.precision import autocast_context, resolve_inference_dtype

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
POSITION_FEATURE_NAMES = tuple(f"{name}.pos" for name in JOINT_NAMES)
EXPECTED_IMAGE_KEYS = ("observation.images.wrist", "observation.images.base")
VARIANT_CHECKPOINTS = {
    "16m": Path(
        "runs/so101-pick-place-dual-cam-resnet18-chunk32/run3/checkpoints/final.pt"
    ),
    "25m": Path(
        "runs/so101-pick-place-dual-cam-resnet18-chunk32-25m/run2/checkpoints/final.pt"
    ),
}


@dataclass(frozen=True)
class Normalization:
    action_mean: np.ndarray
    action_std: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray


def checkpoint_for_variant(variant: str, *, repo_root: Path | None = None) -> Path:
    """Resolve a named local SO-101 checkpoint variant."""

    key = str(variant).strip().lower()
    if key not in VARIANT_CHECKPOINTS:
        raise ValueError(
            f"Unknown variant {variant!r}; choose from {sorted(VARIANT_CHECKPOINTS)}"
        )
    path = VARIANT_CHECKPOINTS[key]
    return (repo_root / path if repo_root is not None else path).resolve()


class PolicyBundle:
    """Frozen model, schema, preprocessing, and normalization in one object."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device: torch.device,
        precision: torch.dtype | None,
        model: torch.nn.Module,
        model_config: ModelConfig,
        normalization: Normalization,
        image_keys: tuple[str, ...],
        image_resize_hw: tuple[int, int],
        image_resize_mode: str,
        weight_source: str,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.precision = precision
        self.model = model
        self.model_config = model_config
        self.normalization = normalization
        self.image_keys = image_keys
        self.camera_names = tuple(
            key.removeprefix("observation.images.") for key in image_keys
        )
        self.image_resize_hw = image_resize_hw
        self.image_resize_mode = image_resize_mode
        self.weight_source = weight_source

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "auto",
        precision: str | None = "auto",
    ) -> PolicyBundle:
        """Load and strictly validate a trained mini-pi0 checkpoint."""

        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        resolved_device = resolve_device(device)
        runtime_dtype = resolve_inference_dtype(
            device=resolved_device, requested=precision
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("model"), dict
        ):
            raise TypeError(
                "Expected a training checkpoint containing a 'model' state dict"
            )
        if checkpoint.get("model_name") != "mini_pi0_fm":
            raise ValueError(
                f"Unsupported checkpoint model: {checkpoint.get('model_name')!r}"
            )

        raw_model_config = checkpoint.get("model_config")
        if not isinstance(raw_model_config, dict):
            raise TypeError("Checkpoint is missing model_config")
        allowed = {item.name for item in fields(ModelConfig)}
        model_config = ModelConfig(
            **{key: value for key, value in raw_model_config.items() if key in allowed}
        )
        if (model_config.action_dim, model_config.prop_dim) != (6, 6):
            raise ValueError(
                "SO-101 inference requires six state/action dimensions; got "
                f"prop={model_config.prop_dim}, action={model_config.action_dim}"
            )
        if model_config.obs_horizon != 1:
            raise ValueError(
                f"This runtime currently requires obs_horizon=1, got {model_config.obs_horizon}"
            )

        robot_config = checkpoint.get("robot_config")
        if not isinstance(robot_config, dict):
            raise TypeError("Checkpoint is missing robot_config")
        image_keys = tuple(str(value) for value in robot_config.get("image_keys", ()))
        if image_keys != EXPECTED_IMAGE_KEYS:
            raise ValueError(
                f"Expected checkpoint cameras in order {EXPECTED_IMAGE_KEYS}, got {image_keys}"
            )
        state_keys = tuple(str(value) for value in robot_config.get("state_keys") or ())
        if state_keys != ("observation.state",):
            raise ValueError(
                f"Expected observation.state proprioception, got {state_keys}"
            )

        normalization_raw = checkpoint.get("normalization")
        if not isinstance(normalization_raw, dict) or not normalization_raw.get(
            "state_normalized", False
        ):
            raise ValueError(
                "Checkpoint must contain normalized state/action statistics"
            )

        def _array(name: str) -> np.ndarray:
            value = np.asarray(normalization_raw.get(name), dtype=np.float32)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError(
                    f"Checkpoint normalization {name!r} must be six finite values"
                )
            return value

        normalization = Normalization(
            action_mean=_array("action_mean"),
            action_std=_array("action_std"),
            state_mean=_array("state_mean"),
            state_std=_array("state_std"),
        )
        if np.any(normalization.action_std <= 0) or np.any(
            normalization.state_std <= 0
        ):
            raise ValueError(
                "Checkpoint normalization standard deviations must be positive"
            )

        resize_hw = tuple(
            int(value) for value in robot_config.get("image_resize_hw") or ()
        )
        if len(resize_hw) != 2 or any(value <= 0 for value in resize_hw):
            raise ValueError(f"Invalid checkpoint image_resize_hw: {resize_hw}")
        model = make_model(model_config)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().requires_grad_(False)
        model.to(resolved_device)
        return cls(
            checkpoint_path=path,
            device=resolved_device,
            precision=runtime_dtype,
            model=model,
            model_config=model_config,
            normalization=normalization,
            image_keys=image_keys,
            image_resize_hw=(resize_hw[0], resize_hw[1]),
            image_resize_mode=str(robot_config.get("image_resize_mode", "letterbox")),
            weight_source=str(checkpoint.get("model_weight_source", "model")),
        )

    @property
    def chunk_size(self) -> int:
        return int(self.model_config.chunk_size)

    @property
    def action_dim(self) -> int:
        return int(self.model_config.action_dim)

    @property
    def parameter_count(self) -> int:
        return count_params(self.model)[0]

    @property
    def precision_name(self) -> str:
        return (
            "float32"
            if self.precision is None
            else str(self.precision).removeprefix("torch.")
        )

    def autocast(self) -> AbstractContextManager[None]:
        return autocast_context(device=self.device, dtype=self.precision)

    def preprocess(
        self,
        cameras: Mapping[str, np.ndarray],
        state: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert RGB HWC frames and calibrated joint positions to model tensors."""

        if tuple(cameras) != self.camera_names:
            raise ValueError(
                f"Camera mapping must preserve trained order {self.camera_names}, got {tuple(cameras)}"
            )
        images: list[torch.Tensor] = []
        for name in self.camera_names:
            frame = np.asarray(cameras[name])
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                raise ValueError(
                    f"Camera {name!r} must be RGB uint8 HWC, got {frame.shape} {frame.dtype}"
                )
            tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
            tensor = resize_image_tensor(
                tensor, self.image_resize_hw, self.image_resize_mode
            )
            images.append(tensor.float().div_(255.0))
        image_tensor = torch.stack(images, dim=0).unsqueeze(0).to(self.device)

        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.shape != (6,) or not np.isfinite(state_array).all():
            raise ValueError(
                f"State must contain six finite joint positions, got {state_array}"
            )
        normalized_state = (
            state_array - self.normalization.state_mean
        ) / self.normalization.state_std
        state_tensor = torch.from_numpy(normalized_state).unsqueeze(0).to(self.device)
        return image_tensor, state_tensor

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(
            self.normalization.action_mean, device=actions.device, dtype=actions.dtype
        )
        std = torch.as_tensor(
            self.normalization.action_std, device=actions.device, dtype=actions.dtype
        )
        return (actions - mean) / std

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(
            self.normalization.action_mean, device=actions.device, dtype=actions.dtype
        )
        std = torch.as_tensor(
            self.normalization.action_std, device=actions.device, dtype=actions.dtype
        )
        return actions * std + mean

    @torch.no_grad()
    def sample(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        *,
        flow_steps: int,
        solver: str = "euler",
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate one normalized action chunk without RTC guidance."""

        with self.autocast():
            return self.model.sample(
                images,
                state,
                n_steps=flow_steps,
                solver=solver,
                initial_noise=initial_noise,
            ).float()

    def warmup(self, *, flow_steps: int = 2) -> None:
        """Materialize backend kernels using synthetic model-shaped inputs."""

        height, width = self.image_resize_hw
        images = torch.zeros(
            1, len(self.camera_names), 3, height, width, device=self.device
        )
        state = torch.zeros(1, self.action_dim, device=self.device)
        self.sample(images, state, flow_steps=max(1, flow_steps), solver="euler")
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
