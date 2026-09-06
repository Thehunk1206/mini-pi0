"""LeRobot Real-Time Chunking adapter for mini-pi0 flow matching."""

from __future__ import annotations

import torch
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc import RTCConfig, RTCProcessor

from .config import RTCInferenceConfig
from .policy_bundle import PolicyBundle


def make_lerobot_rtc_config(config: RTCInferenceConfig) -> RTCConfig:
    """Translate local CLI settings to LeRobot's official RTC configuration."""

    try:
        schedule = RTCAttentionSchedule[
            str(config.prefix_attention_schedule).strip().upper()
        ]
    except KeyError as exc:
        choices = ", ".join(item.name for item in RTCAttentionSchedule)
        raise ValueError(
            f"Unknown RTC attention schedule; choose one of {choices}"
        ) from exc
    return RTCConfig(
        enabled=bool(config.enabled),
        prefix_attention_schedule=schedule,
        max_guidance_weight=float(config.max_guidance_weight),
        execution_horizon=int(config.execution_horizon),
    )


class MiniPi0RTCPolicy:
    """Sample mini-pi0 chunks with optional official RTC prefix guidance."""

    def __init__(self, bundle: PolicyBundle, config: RTCInferenceConfig) -> None:
        self.bundle = bundle
        self.config = config
        self.rtc_config = make_lerobot_rtc_config(config)
        self.processor = RTCProcessor(self.rtc_config)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.seed))
        self._fixed_noise_cpu = torch.randn(
            1,
            bundle.chunk_size,
            bundle.action_dim,
            generator=generator,
            dtype=torch.float32,
        )

    def reset(self) -> None:
        self.processor.reset_tracker()

    def sample_normalized(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        *,
        previous_leftover: torch.Tensor | None,
        inference_delay: int,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate a normalized chunk using the model's noise-to-data convention."""

        if initial_noise is None and self.config.fixed_noise:
            initial_noise = self._fixed_noise_cpu.to(
                device=self.bundle.device,
                dtype=images.dtype,
            )
        if not self.config.enabled or previous_leftover is None:
            return self.bundle.sample(
                images,
                state,
                flow_steps=self.config.flow_steps,
                solver=self.config.solver,
                initial_noise=initial_noise,
            )

        model = self.bundle.model
        batch = int(images.shape[0])
        expected = (batch, self.bundle.chunk_size, self.bundle.action_dim)
        if initial_noise is None:
            actions = torch.randn(
                expected, device=self.bundle.device, dtype=images.dtype
            )
        else:
            if tuple(initial_noise.shape) != expected:
                raise ValueError(
                    f"Expected initial noise {expected}, got {tuple(initial_noise.shape)}"
                )
            actions = initial_noise.to(device=self.bundle.device, dtype=images.dtype)
        previous = previous_leftover.to(device=self.bundle.device, dtype=actions.dtype)
        if previous.ndim == 2:
            previous = previous.unsqueeze(0)

        with torch.no_grad(), self.bundle.autocast():
            conditioning = model.encode_conditioning(images, state).detach()
        time_grid = torch.linspace(
            0.0,
            1.0,
            self.config.flow_steps + 1,
            device=self.bundle.device,
            dtype=actions.dtype,
        )
        for index in range(self.config.flow_steps):
            model_time = time_grid[index]
            dt = time_grid[index + 1] - model_time

            def negative_velocity(
                value: torch.Tensor,
                current_model_time: torch.Tensor = model_time,
            ) -> torch.Tensor:
                return -model.flow_velocity(value, current_model_time, conditioning)

            with self.bundle.autocast():
                guided_negative_velocity = self.processor.denoise_step(
                    x_t=actions,
                    prev_chunk_left_over=previous,
                    inference_delay=max(0, int(inference_delay)),
                    time=1.0 - model_time,
                    original_denoise_step_partial=negative_velocity,
                    execution_horizon=self.config.execution_horizon,
                )
            actions = (actions - dt * guided_negative_velocity).detach()
        return actions.float()
