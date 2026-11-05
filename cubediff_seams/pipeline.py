"""Frozen-weight baseline; conditional and unconditional calls both keep six faces."""

from dataclasses import dataclass
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.utils import BaseOutput
from .geometry import extra_channels, postprocess
from .modules import patch_attention, patch_vae_norms


@dataclass
class PanoramaOutput(BaseOutput):
    faces: np.ndarray
    faces_cropped: np.ndarray
    equirectangular: np.ndarray


class PanoramaPipeline(StableDiffusionPipeline):
    @classmethod
    def from_pretrained(cls, checkpoint, **kwargs):
        pipe = super().from_pretrained(checkpoint, **kwargs)
        if pipe.unet.config.in_channels != 7 or not isinstance(pipe.scheduler, DDIMScheduler) or pipe.scheduler.config.prediction_type != 'epsilon':
            raise ValueError('Expected a trained seven-channel DDIM epsilon checkpoint')
        patch_attention(pipe.unet)
        patch_vae_norms(pipe.vae)
        return pipe

    @torch.no_grad()
    def __call__(self, prompt, *, conditioning_image, num_inference_steps=50,
                 generator=None, cfg_scale=3.5):
        device, dtype = self.unet.device, self.unet.dtype
        size = self.unet.config.sample_size
        expected = size * self.vae_scale_factor
        if conditioning_image.shape != (3, expected, expected):
            raise ValueError(f'Expected a (3, {expected}, {expected}) conditioning image')
        if not torch.isfinite(conditioning_image).all() or conditioning_image.min() < -1 or conditioning_image.max() > 1:
            raise ValueError('Conditioning image must be finite and normalized to [-1, 1]')
        prompts = [prompt] * 6
        embeddings = []
        for captions in (prompts, [''] * 6):
            inputs = self.tokenizer(captions, max_length=self.tokenizer.model_max_length,
                                    padding='max_length', truncation=True, return_tensors='pt')
            embeddings.append(self.text_encoder(inputs.input_ids.to(self.text_encoder.device))[0].to(device=device, dtype=dtype))
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = torch.randn((6, 4, size, size), device=device, dtype=dtype, generator=generator)
        latents *= self.scheduler.init_noise_sigma
        static = extra_channels(size, device, dtype)
        condition = conditioning_image[None].to(device=self.vae.device, dtype=self.vae.dtype)
        reference = (self.vae.encode(condition).latent_dist.mean[0] * self.vae.config.scaling_factor).to(device=device, dtype=dtype)
        for timestep in self.scheduler.timesteps:
            latents[0] = reference
            inputs = torch.cat((self.scheduler.scale_model_input(latents, timestep), static), dim=1)
            conditional = self.unet(inputs, timestep, encoder_hidden_states=embeddings[0]).sample
            unconditional = self.unet(inputs, timestep, encoder_hidden_states=embeddings[1],
                                      cross_attention_kwargs={'front_face_drop': True}).sample
            prediction = unconditional + cfg_scale * (conditional - unconditional)
            latents[1:] = self.scheduler.step(prediction[1:], timestep, latents[1:]).prev_sample
        decoded = self.vae.decode((latents / self.vae.config.scaling_factor).to(device=self.vae.device, dtype=self.vae.dtype)).sample
        if not torch.isfinite(decoded).all():
            raise FloatingPointError('Nonfinite VAE output')
        faces, cropped, panorama = postprocess((decoded / 2 + 0.5).clamp(0, 1))
        return PanoramaOutput(faces=faces, faces_cropped=cropped, equirectangular=panorama)
