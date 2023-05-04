# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

import jax
import jax.numpy as jnp
import flax.linen as nn

from einops import rearrange


class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x


class FlaxUpsample3D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32
    use_conv: bool=False
    use_conv_transpose: bool = False
    name: str = "conv"

    def setup(self):
        if self.name == "conv":
            self.conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)
        else:
            self.Conv2d_0 = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)

    def __call__(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        hidden_states = self.conv(hidden_states)
        return hidden_states

class FlaxDownsample3D(nn.Module):
    channels: int
    dtype: jnp.dtype = jnp.float32
    out_channels: int = None
    use_conv: bool = False
    padding: int = 1
    name: str = "conv"

    def setup(self):
        stride = 2
        if self.use_conv:
            conv = InflatedConv3d(self.channels,
                                  self.out_channels,
                                  3,
                                  stride=stride,
                                  padding=self.padding)
        else:
            raise NotImplementedError

        if self.name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif self.name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def __call__(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            raise NotImplementedError

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states

class FlaxResnetBlock3D(nn.Module):
    in_channels: int
    out_channels: int = None
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    conv_shortcut: bool = False
    temb_channels: int = 512
    groups: int = 32
    groups_out=None
    pre_norm: bool =True
    eps: float=1e-6
    non_linearity: str="swish"
    time_embedding_norm: str="default"
    output_scale_factor: float=1.0
    use_in_shortcut=None
    use_nin_shortcut: bool = None

    def setup(self):

        self.norm1 = nn.GroupNorm(num_groups=self.groups, group_size=self.in_channels, eps=self.eps, affine=True)

        self.conv1 = InflatedConv3d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        if self.temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = self.out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = self.out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj =  nn.Dense(time_emb_proj_out_channels, dtype=self.dtype)
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=self.groups_out, group_size=self.out_channels, eps=self.eps, affine=True)
        self.dropout = nn.Dropout(self.dropout)
        self.conv2 = InflatedConv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        if self.non_linearity == "swish":
            self.nonlinearity = nn.swish
        elif self.non_linearity == "mish":
            self.nonlinearity = Mish()
        elif self.non_linearity == "silu":
            self.nonlinearity = jax.nn.silu

        self.use_in_shortcut = self.in_channels != self.out_channels if self.use_in_shortcut is None else self.use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedConv3d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, input_tensor, temb, deterministic=True):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = jnp.split(temb, 2, axis=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states, deterministic)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class Mish(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * jnp.tanh(jax.nn.softplus(hidden_states))