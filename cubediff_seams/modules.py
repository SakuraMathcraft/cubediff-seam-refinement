"""Six-face attention and VAE normalization for the audited SD 1.5 architecture."""

import torch
from torch import nn
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.transformers.transformer_2d import Transformer2DModel


class JointTransformerBlock(BasicTransformerBlock):
    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, timestep=None, cross_attention_kwargs=None,
                class_labels=None, added_cond_kwargs=None):
        if attention_mask is not None or class_labels is not None or added_cond_kwargs:
            raise ValueError('Unsupported conditioning for this fixed architecture')
        options = dict(cross_attention_kwargs or {})
        drop_front = options.pop('front_face_drop', False)
        if options:
            raise ValueError(f'Unknown attention options: {sorted(options)}')
        count, tokens, channels = hidden_states.shape
        if count % 6:
            raise ValueError('Attention requires complete groups of six faces')
        groups = count // 6
        joint = self.norm1(hidden_states).reshape(groups, 6 * tokens, channels)
        mask = None
        if drop_front:
            mask = torch.ones((groups, 1, 1, 6 * tokens), dtype=torch.bool, device=joint.device)
            mask[..., :tokens] = False
        attended = self.attn1(joint, attention_mask=mask)
        hidden_states = hidden_states + attended.reshape(count, tokens, channels)
        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states), encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        return hidden_states + self.ff(self.norm3(hidden_states))


def patch_attention(unet):
    for module in unet.modules():
        if not isinstance(module, Transformer2DModel):
            continue
        for index, block in enumerate(module.transformer_blocks):
            if type(block) is not BasicTransformerBlock or block.norm_type != 'layer_norm' or block.only_cross_attention or block.double_self_attention or block.pos_embed is not None:
                raise ValueError('Checkpoint must use the audited SD 1.5 transformer architecture')
            replacement = JointTransformerBlock(
                dim=block.dim, num_attention_heads=block.num_attention_heads,
                attention_head_dim=block.attention_head_dim, dropout=block.dropout,
                cross_attention_dim=block.cross_attention_dim,
                activation_fn=block.activation_fn, attention_bias=block.attention_bias,
                norm_elementwise_affine=block.norm_elementwise_affine,
                norm_type='layer_norm', norm_eps=block.norm1.eps,
                upcast_attention=block.attn1.upcast_attention,
            )
            parameter = next(block.parameters())
            replacement.to(device=parameter.device, dtype=parameter.dtype)
            replacement.load_state_dict(block.state_dict(), strict=True)
            replacement.train(block.training)
            replacement.attn1.set_processor(AttnProcessor2_0())
            module.transformer_blocks[index] = replacement


class JointGroupNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, values):
        if values.ndim not in (3, 4):
            raise ValueError('VAE normalization requires spatial features')
        count, channels, *spatial = values.shape
        if count == 1:
            return self.norm(values)
        if count % 6:
            raise ValueError('VAE decoding requires complete groups of six faces')
        joint = values.reshape(count // 6, 6, channels, *spatial).transpose(1, 2)
        normalized = self.norm(joint.reshape(count // 6, channels, -1))
        return normalized.reshape(count // 6, channels, 6, *spatial).transpose(1, 2).reshape_as(values)


def patch_vae_norms(root):
    for name, child in list(root.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(root, name, JointGroupNorm(child))
        elif isinstance(child, JointGroupNorm):
            raise ValueError('VAE has already been patched')
        else:
            patch_vae_norms(child)
