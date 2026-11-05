"""Single epsilon API wired to fixed DDIM inference, without candidate isolation."""

from contextlib import contextmanager

import torch
from diffusers import DDIMScheduler
from .method_api import MethodContext
from .boundary import MethodBoundary


class SamplingStep:
    def __init__(self, pipeline, conditioning_image, factory, seed):
        scheduler = pipeline.scheduler
        if (not isinstance(scheduler, DDIMScheduler) or scheduler.config.prediction_type != 'epsilon'
                or scheduler.config.clip_sample or scheduler.config.thresholding
                or pipeline.unet.config.sample_size != 64):
            raise ValueError('Requires fixed unclipped 64-pixel epsilon DDIM')
        scheduler.set_timesteps(50, device=pipeline.unet.device)
        with torch.no_grad():
            image = conditioning_image[None].to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)
            front = (pipeline.vae.encode(image).latent_dist.mean * pipeline.vae.config.scaling_factor).to(
                device=pipeline.unet.device, dtype=pipeline.unet.dtype).float()
        self.context = MethodContext(front, scheduler.alphas_cumprod.detach().clone(),
            tuple(int(t) for t in scheduler.timesteps), scheduler.final_alpha_cumprod.detach().clone(),
            seed=seed)
        self.boundary = MethodBoundary(factory, self.context)
        self.original_step = scheduler.step

    @torch.no_grad()
    def step(self, epsilon, timestep, sample):
        index = self.boundary.calls
        with deterministic_update():
            updated = self.boundary.update(epsilon, sample, int(timestep), index)
        return self.original_step(updated, timestep, sample)

    def finish(self):
        self.boundary.finish()


@contextmanager
def deterministic_update():
    enabled = torch.are_deterministic_algorithms_enabled()
    warning = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warning)
