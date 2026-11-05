"""Generate a panorama from a fingerprinted local checkpoint, without network access."""

import argparse
import hashlib
import json
import time
from functools import partial
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .geometry import FACE_NAMES
from .measure import measure
from .method import ScheduledGraphMethod


class IdentityMethod:
    def __init__(self, context):
        pass

    def update(self, epsilon, sample, timestep, step_index):
        return epsilon


def verify_checkpoint(directory):
    manifest = json.loads(files('cubediff_seams').joinpath('checkpoint_sha256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        with (directory / name).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Checkpoint fingerprint mismatch: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--method', choices=('baseline', 'refined', 'expanded'), default='refined')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('Generation requires a CUDA-enabled PyTorch installation')
    torch.set_num_threads(8)
    with Image.open(args.image) as image:
        if image.size != (512, 512):
            raise ValueError('Expected a 512×512 conditioning image with a 95° field of view')
        pixels = np.asarray(image.convert('RGB')).copy()
    condition = torch.from_numpy(pixels).permute(2, 0, 1).float() / 127.5 - 1
    print('Verifying local checkpoint…', flush=True)
    verify_checkpoint(args.checkpoint)
    from .pipeline import PanoramaPipeline
    from .sampling import SamplingStep

    pipe = PanoramaPipeline.from_pretrained(args.checkpoint, local_files_only=True,
                                           use_safetensors=True, torch_dtype=torch.float32)
    pipe.unet.to(device='cuda', dtype=torch.float16).eval()
    pipe.vae.to(device='cpu', dtype=torch.float32).eval()
    pipe.text_encoder.to(device='cpu', dtype=torch.float32).eval()
    factory = IdentityMethod if args.method == 'baseline' else partial(
        ScheduledGraphMethod, trust_fraction=.35 if args.method == 'expanded' else .25)
    start = time.perf_counter()
    step = SamplingStep(pipe, condition, factory, args.seed)
    calls = 0

    def count_unet(module, inputs, output):
        nonlocal calls
        if inputs[0].shape != (6, 7, 64, 64):
            raise ValueError('Unexpected UNet input shape')
        calls += 1

    handle = pipe.unet.register_forward_hook(count_unet)
    pipe.scheduler.step = step.step
    try:
        print(f'Generating {args.method} panorama (50 steps)…', flush=True)
        result = pipe(args.prompt, conditioning_image=condition,
                      generator=torch.Generator(device='cuda').manual_seed(args.seed))
        step.finish()
        torch.cuda.synchronize()
    finally:
        pipe.scheduler.step = step.original_step
        handle.remove()
    if calls != 100:
        raise ValueError(f'Expected 100 UNet forwards; got {calls}')
    elapsed = time.perf_counter() - start
    args.output.mkdir(parents=True, exist_ok=False)
    for name, face in zip(FACE_NAMES, result.faces):
        Image.fromarray(face).save(args.output / f'{name}_95.png')
    Image.fromarray(result.equirectangular).save(args.output / 'panorama.png')
    metrics = measure(args.output)
    metrics.update(method=args.method, seed=args.seed, prompt=args.prompt, ddim_steps=step.boundary.calls,
                   unet_forwards=calls, setup_and_inference_seconds=elapsed)
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"Overlap RGB L1: {metrics['overlap_band']['mean_rgb_l1']:.6f}")


if __name__ == '__main__':
    main()
