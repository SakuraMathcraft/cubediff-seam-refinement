"""Draft single submission interface; no model or scheduler objects are exposed."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MethodContext:
    """Per-case data copies supplied by the trusted sampling driver.

    front_latent: float32 [1, 4, 64, 64], fixed encoded condition.
    alphas_cumprod: float32 [1000], the frozen DDIM schedule.
    timesteps: the 50 descending inference timesteps.
    final_alpha_cumprod: scalar float32 for the final DDIM update.
    A frozen dataclass does not make its tensor storage immutable. The driver
    owns independent copies and never uses candidate-mutated storage.
    """

    front_latent: torch.Tensor
    alphas_cumprod: torch.Tensor
    timesteps: tuple[int, ...]
    final_alpha_cumprod: torch.Tensor
    seed: int
