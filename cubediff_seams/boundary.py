"""Local interface checks; this in-process harness is not a candidate sandbox."""

from dataclasses import replace

import torch
from .method_api import MethodContext


class MethodBoundary:
    def __init__(self, factory, context: MethodContext):
        if (context.front_latent.shape != (1,4,64,64)
                or context.front_latent.dtype != torch.float32
                or context.alphas_cumprod.shape != (1000,)
                or context.alphas_cumprod.dtype != torch.float32
                or context.final_alpha_cumprod.shape != torch.Size([])
                or context.final_alpha_cumprod.dtype != torch.float32
                or len(context.timesteps) != 50
                or tuple(sorted(set(context.timesteps), reverse=True)) != context.timesteps
                or not all(type(t) is int and 0 <= t < 1000 for t in context.timesteps)):
            raise ValueError('Invalid fixed sampling context')
        for value in (context.front_latent, context.alphas_cumprod, context.final_alpha_cumprod):
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite context')
        if not (0 < context.alphas_cumprod).all() or not (context.alphas_cumprod <= 1).all():
            raise ValueError('Invalid DDIM alpha')
        if not 0 < context.final_alpha_cumprod <= 1:
            raise ValueError('Invalid final DDIM alpha')
        self._timesteps = context.timesteps
        self._device = context.front_latent.device
        self.calls = 0
        supplied = replace(context, front_latent=context.front_latent.detach().clone(),
            alphas_cumprod=context.alphas_cumprod.detach().clone(),
            final_alpha_cumprod=context.final_alpha_cumprod.detach().clone())
        self._method = factory(supplied)

    @torch.no_grad()
    def update(self, epsilon, noisy_targets, timestep, step_index):
        if (type(step_index) is not int or type(timestep) is not int or step_index != self.calls
                or self.calls >= 50 or timestep != self._timesteps[self.calls]):
            raise ValueError('Incorrect method invocation order')
        if (not isinstance(epsilon, torch.Tensor) or not isinstance(noisy_targets, torch.Tensor)
                or epsilon.shape != (5,4,64,64) or noisy_targets.shape != epsilon.shape
                or epsilon.dtype not in (torch.float16, torch.float32)
                or epsilon.device != self._device
                or noisy_targets.dtype != epsilon.dtype or noisy_targets.device != epsilon.device
                or not torch.isfinite(epsilon).all() or not torch.isfinite(noisy_targets).all()):
            raise ValueError('Invalid method input')
        result = self._method.update(epsilon.detach().clone(), noisy_targets.detach().clone(), timestep, step_index)
        if (not isinstance(result, torch.Tensor) or result.shape != epsilon.shape
                or result.dtype != epsilon.dtype or result.device != epsilon.device
                or not torch.isfinite(result).all()):
            raise ValueError('Invalid method output')
        self.calls += 1
        return result.detach().clone()

    def finish(self):
        if self.calls != 50:
            raise ValueError('Incomplete method trajectory')
