"""Expert prototype: texture-weighted, anchored all-edge latent graph solve."""

import math
import torch
from torch.nn import functional as F
from .geometry import BASES, FOV_DEG, face_rays


class OverlapGraph:
    """Bilinear directed ray constraints with an exact discrete transpose.

    All 24 directed adjacencies enter the objective. Front-face variables are
    fixed; other variables touched by a constraint form the update support.
    Geometry of latent channels remains an empirical approximation.
    """

    def __init__(self, size, device):
        if size < 4:
            raise ValueError('Need a nontrivial square plane')
        self.size = size
        extent = math.tan(math.radians(FOV_DEG / 2))
        rays, bases = face_rays(size), BASES
        indices, coefficients, weights = [], [], []
        pixels = torch.arange(size * size).reshape(size, size)
        for target in range(6):
            for source in range(6):
                if torch.dot(bases[target, 0], bases[source, 0]) != 0:
                    continue
                normal, right, down = bases[source]
                depth = rays[target] @ normal
                denominator = depth.clamp_min(1e-6)
                grid = torch.stack((rays[target] @ right / denominator,
                                    rays[target] @ down / denominator), -1) / extent
                margin = 1 - grid.abs().amax(-1)
                visible = (depth > 0) & (margin > 1e-6)
                coordinates = (grid[visible] + 1) * ((size - 1) / 2)
                x, y = coordinates.unbind(-1)
                x0, y0 = x.floor().long(), y.floor().long()
                x1, y1 = (x0 + 1).clamp_max(size - 1), (y0 + 1).clamp_max(size - 1)
                dx, dy = x - x0, y - y0
                locations = torch.stack((pixels[visible] + target * size * size,
                    y0 * size + x0 + source * size * size,
                    y0 * size + x1 + source * size * size,
                    y1 * size + x0 + source * size * size,
                    y1 * size + x1 + source * size * size), -1)
                values = torch.stack((torch.ones_like(dx), -(1-dx)*(1-dy),
                                      -dx*(1-dy), -(1-dx)*dy, -dx*dy), -1)
                indices.append(locations)
                coefficients.append(values)
                weights.append((margin[visible] / (1 - 1 / extent**2)).clamp(0, 1))
        self.indices = torch.cat(indices).to(device)
        self.coefficients = torch.cat(coefficients).to(device)
        self.geometric_weights = torch.cat(weights).to(device)
        self.variable_mask = torch.zeros(6 * size * size, device=device, dtype=torch.bool)
        self.variable_mask[self.indices[self.coefficients != 0]] = True
        self.variable_mask[:size * size] = False

    def residual(self, values):
        return (values[self.indices] * self.coefficients[..., None]).sum(1)

    def transpose(self, residual):
        values = torch.zeros(6 * self.size**2, residual.shape[-1], device=residual.device, dtype=residual.dtype)
        values.index_add_(0, self.indices.flatten(),
                          (residual[:, None] * self.coefficients[..., None]).flatten(0, 1))
        return values * self.variable_mask[:, None]

    def flatten(self, clean):
        if clean.ndim != 4 or clean.shape[0] != 6 or clean.shape[-2:] != (self.size, self.size):
            raise ValueError('Expected six matching square planes')
        return clean.float().permute(0, 2, 3, 1).reshape(6 * self.size**2, -1)

    def confidence(self, clean):
        smooth = F.avg_pool2d(F.pad(clean.float(), (1, 1, 1, 1), mode='replicate'), 3, stride=1)
        high = self.flatten(clean.float() - smooth)
        sampled = high[self.indices]
        target = sampled[:, 0]
        source = -(sampled[:, 1:] * self.coefficients[:, 1:, None]).sum(1)
        scale = (target.square().mean(-1) + source.square().mean(-1)).clamp_min(0.01)
        disagreement = (target - source).square().mean(-1)
        return self.geometric_weights / (1 + disagreement / scale)


class LocalTrustGraph:
    """Reuse the audited ray graph; impose each pixel's own convex RMS ball."""

    def __init__(self, geometry):
        self.geometry = geometry
        self.size = geometry.size

    @torch.no_grad()
    def solve(self, clean, coupling, iterations=12, trust_fraction=.25):
        if not 0 <= coupling <= 1 or iterations < 1 or not 0 < trust_fraction <= 1:
            raise ValueError('Invalid local graph settings')
        graph = self.geometry
        anchor = graph.flatten(clean)
        if not torch.isfinite(anchor).all():
            raise ValueError('Nonfinite clean prediction')
        if coupling == 0:
            return clean.clone(), {'iterations':0,'coupling':0.0}
        weights = graph.confidence(clean)
        face_limits = trust_fraction * anchor.reshape(6,self.size**2,-1).flatten(1).std(1,unbiased=False).clamp_min(.1)
        limits = face_limits[:,None,None].expand(6,self.size**2,1).reshape(-1,1)

        def project(delta):
            norm = delta.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12)
            return delta * (limits/norm).clamp_max(1) * graph.variable_mask[:,None]

        def gradient(delta):
            return delta + coupling * graph.transpose(weights[:,None] * graph.residual(anchor+delta))

        def objective(delta):
            return delta.square().sum() + coupling * (weights[:,None]*graph.residual(anchor+delta).square()).sum()

        # Absolute row-sum upper bound of I + lambda A_t^T W A_t.
        # Symmetry makes it a valid spectral bound; no score-dependent step search.
        active_coefficients = graph.coefficients.abs() * graph.variable_mask[graph.indices]
        row_norm = active_coefficients.sum(1)
        bounds = torch.zeros(len(anchor),device=anchor.device)
        bounds.index_add_(0,graph.indices.flatten(),
                         (weights[:,None]*active_coefficients*row_norm[:,None]).flatten())
        step = 1 / (1 + coupling*bounds.max())
        delta,extrapolated = torch.zeros_like(anchor),torch.zeros_like(anchor)
        before = objective(delta)
        previous_objective = before
        momentum, restarts = 1.0,0
        maximum_increase = 0.0
        for index in range(iterations):
            next_delta = project(extrapolated-step*gradient(extrapolated))
            energy = objective(next_delta)
            if energy > previous_objective:
                restarts += 1
                momentum = 1.0
                next_delta = project(delta-step*gradient(delta))
                energy = objective(next_delta)
            if not torch.isfinite(energy) or energy > previous_objective*(1+1e-5)+1e-7:
                raise FloatingPointError('Local constrained objective increased')
            maximum_increase = max(maximum_increase,float(energy-previous_objective))
            next_momentum = (1+math.sqrt(1+4*momentum**2))/2
            extrapolated = next_delta+(momentum-1)/next_momentum*(next_delta-delta)
            delta, momentum, previous_objective = next_delta,next_momentum,energy
        projected_step = project(delta-step*gradient(delta))-delta
        corrected = (anchor+delta).reshape(6,self.size,self.size,-1).permute(0,3,1,2).to(clean.dtype)
        norm = delta.square().mean(-1).sqrt()
        active = graph.variable_mask
        if not torch.isfinite(corrected).all():
            raise FloatingPointError('Nonfinite corrected prediction')
        return corrected, {'solver':'monotone_accelerated_local_projection','iterations':iterations,
            'coupling':float(coupling),'objective_before':float(before),'objective_after':float(previous_objective),
            'step_size':float(step),'restarts':restarts,'max_iteration_objective_increase':maximum_increase,
            'relative_projected_step':float(projected_step.norm()/delta.norm().clamp_min(1e-12)),
            'max_pixel_limit_ratio':float((norm[:,None]/limits)[active].max()),
            'active_limit_fraction':float((norm[:,None] >= limits*(1-1e-5))[active].float().mean())}
