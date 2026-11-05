"""Parameterized temporal graph updates for a predeclared development ablation."""

import torch
from .graph import LocalTrustGraph, OverlapGraph


class ScheduledGraphMethod:
    def __init__(self, context, trust_fraction=.25):
        if not 0 < trust_fraction <= 1:
            raise ValueError("Invalid pixel trust fraction")
        self.trust_fraction = trust_fraction
        self.context = context
        self.graph = LocalTrustGraph(OverlapGraph(64, context.front_latent.device))
        self.mask = self.graph.geometry.variable_mask.reshape(6,64,64)[1:,None]
        self.shadow = None
        self.diagnostics = []

    @torch.no_grad()
    def update(self, epsilon, noisy_targets, timestep, step_index):
        if step_index < 35:
            return epsilon
        current = noisy_targets.float()
        if self.shadow is None:
            self.shadow = current.clone()
        shadow = torch.where(self.mask, self.shadow, current)
        alpha = self.context.alphas_cumprod[timestep].to(current.device)
        beta = 1-alpha
        shadow_clean = (shadow-beta.sqrt()*epsilon.float())/alpha.sqrt()
        corrected, record = self.graph.solve(torch.cat((self.context.front_latent,shadow_clean)),
            (step_index-34)/15, iterations=12, trust_fraction=self.trust_fraction)
        updated = ((current-alpha.sqrt()*corrected[1:])/beta.sqrt()).to(epsilon.dtype)
        updated = torch.where(self.mask, updated, epsilon)
        previous_alpha = (self.context.alphas_cumprod[self.context.timesteps[step_index+1]]
            if step_index < 49 else self.context.final_alpha_cumprod).to(current.device)
        self.shadow = previous_alpha.sqrt()*shadow_clean+(1-previous_alpha).sqrt()*epsilon.float()
        if not torch.isfinite(self.shadow).all():
            raise FloatingPointError('Nonfinite temporal reference')
        self.diagnostics.append(record)
        return updated
