"""Audited face order and pinhole geometry; metrics here are diagnostic only."""

import math
import numpy as np
import torch
import torch.nn.functional as F

FACE_NAMES = ('front', 'back', 'left', 'right', 'top', 'bottom')
FOV_DEG = 95.0
# Per face: forward normal, image-right basis, image-down basis.
BASES = torch.tensor([
    [[0, 0, 1], [1, 0, 0], [0, -1, 0]],
    [[0, 0, -1], [-1, 0, 0], [0, -1, 0]],
    [[-1, 0, 0], [0, 0, 1], [0, -1, 0]],
    [[1, 0, 0], [0, 0, -1], [0, -1, 0]],
    [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, -1]],
], dtype=torch.float32)


def face_rays(size):
    extent = math.tan(math.radians(FOV_DEG / 2))
    axis = torch.linspace(-extent, extent, size)
    down, right = torch.meshgrid(axis, axis, indexing='ij')
    return (BASES[:, 0, None, None] + right[None, ..., None] * BASES[:, 1, None, None]
            + down[None, ..., None] * BASES[:, 2, None, None])


def extra_channels(size, device, dtype):
    rays = face_rays(size)
    x, y, z = rays.unbind(-1)
    longitude = (torch.atan2(x, z) / math.pi + 1) / 2
    latitude = (torch.atan2(y, torch.sqrt(x.square() + z.square())) / math.pi + 1) / 2
    front = torch.zeros_like(longitude)
    front[0] = 1
    return torch.stack((longitude, latitude, front), dim=1).to(device=device, dtype=dtype)


def _overlap_errors(faces, samples, band_samples):
    if faces.ndim != 4 or faces.shape[:2] != (6, 3) or faces.shape[-1] != faces.shape[-2]:
        raise ValueError('Expected six square RGB faces')
    if not torch.isfinite(faces).all() or faces.min() < 0 or faces.max() > 1:
        raise ValueError('Expected finite RGB values in [0, 1]')
    if samples < 2 or band_samples < 1:
        raise ValueError('Need at least two along-edge samples and one band sample')
    extent = math.tan(math.radians(FOV_DEG / 2))
    bases = BASES.to(device=faces.device, dtype=faces.dtype)
    along = torch.linspace(-0.9, 0.9, samples, device=faces.device, dtype=faces.dtype)
    # At r=n1+n2+offset*(n1-n2), both projected coordinates must lie in the 95-degree face.
    width = 0.95 * (extent - 1) / (extent + 1)
    offsets = (torch.zeros(1, device=faces.device, dtype=faces.dtype) if band_samples == 1
               else torch.linspace(-width, width, band_samples, device=faces.device, dtype=faces.dtype))
    errors = {}
    for first in range(6):
        for second in range(first + 1, 6):
            n1, n2 = bases[first, 0], bases[second, 0]
            if torch.dot(n1, n2) != 0:
                continue
            tangent = torch.linalg.cross(n1, n2)
            rays = (n1 + n2 + along[None, :, None] * tangent
                    + offsets[:, None, None] * (n1 - n2))
            rendered = []
            for index in (first, second):
                normal, right, down = bases[index]
                denominator = rays @ normal
                grid = torch.stack((rays @ right / denominator, rays @ down / denominator), dim=-1) / extent
                rendered.append(F.grid_sample(faces[index:index + 1], grid[None], align_corners=True, mode='bilinear'))
            errors[f'{FACE_NAMES[first]}:{FACE_NAMES[second]}'] = (rendered[0] - rendered[1]).abs().mean().item()
    return {'mean_rgb_l1': sum(errors.values()) / len(errors), 'edges': errors}


def overlap_band_diagnostic(faces, samples=128, band_samples=17):
    """Mismatch across 95% of the shared overlap width, excluding cube corners.

    Uses uniform ray-parameter samples, not a solid-angle integral. Diagnostic only.
    """
    return _overlap_errors(faces, samples, band_samples)


def postprocess(images):
    import py360convert

    arrays = images.detach().cpu().float().permute(0, 2, 3, 1).numpy()
    size = arrays.shape[1]
    cropped_size = round(size / math.tan(math.radians(FOV_DEG / 2)))
    start = (size - cropped_size) // 2
    cropped = arrays[:, start:start + cropped_size, start:start + cropped_size]
    cube = dict(zip(('F', 'B', 'L', 'R', 'U', 'D'), cropped))
    panorama = py360convert.c2e(cube, h=1024, w=2048, cube_format='dict')
    return tuple((np.clip(array, 0, 1) * 255).astype(np.uint8) for array in (arrays, cropped, panorama))
