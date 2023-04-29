# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import math

import flax.linen as nn
import jax
import jax.numpy as jnp

# from diffusers.models.attention_flax import FlaxBasicTransformerBlock
from diffusers.models.attention_flax import FlaxFeedForward, jax_memory_efficient_attention

def rearrange_3(array, f): 
    F, D, C = array.shape
    return jnp.reshape(array, (F // f, f, D, C))

def rearrange_4(array):
    B, F, D, C = array.shape
    return jnp.reshape(array, (B * F, D, C))

class FlaxCrossFrameAttention(nn.Module):
    r"""
    A Flax multi-head attention module as described in: https://arxiv.org/abs/1706.03762

    Parameters:
        query_dim (:obj:`int`):
            Input hidden states dimension
        heads (:obj:`int`, *optional*, defaults to 8):
            Number of heads
        dim_head (:obj:`int`, *optional*, defaults to 64):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        batch_size: The number that represents actual batch size, other than the frames.
            For example, using calling unet with a single prompt and num_images_per_prompt=1, batch_size should be
            equal to 2, due to classifier-free guidance.

    """
    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    use_memory_efficient_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    batch_size : int = 2

    def setup(self):
        inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head**-0.5

        # Weights were exported with old names {to_q, to_k, to_v, to_out}
        self.query = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_q")
        self.key = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_k")
        self.value = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_v")

        self.add_k_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype)
        self.add_v_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype)

        self.proj_attn = nn.Dense(self.query_dim, dtype=self.dtype, name="to_out_0")

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def __call__(self, hidden_states, context=None, deterministic=True):
        is_cross_attention = context is not None
        context = hidden_states if context is None else context
        query_proj = self.query(hidden_states)
        key_proj = self.key(context)
        value_proj = self.value(context)

        # Sparse Attention
        if not is_cross_attention:
            video_length = 1 if key_proj.shape[0] < self.batch_size else key_proj.shape[0] // self.batch_size
            first_frame_index = [0] * video_length

            # rearrange keys to have batch and frames in the 1st and 2nd dims respectively
            key_proj = rearrange_3(key_proj, video_length)
            key_proj = key_proj[:, first_frame_index]
            # rearrange values to have batch and frames in the 1st and 2nd dims respectively
            value_proj = rearrange_3(value_proj, video_length)
            value_proj = value_proj[:, first_frame_index]

            # rearrange back to original shape
            key_proj = rearrange_4(key_proj)
            value_proj = rearrange_4(value_proj)

        query_states = self.reshape_heads_to_batch_dim(query_proj)
        key_states = self.reshape_heads_to_batch_dim(key_proj)
        value_states = self.reshape_heads_to_batch_dim(value_proj)

        if self.use_memory_efficient_attention:
            query_states = query_states.transpose(1, 0, 2)
            key_states = key_states.transpose(1, 0, 2)
            value_states = value_states.transpose(1, 0, 2)

            # this if statement create a chunk size for each layer of the unet
            # the chunk size is equal to the query_length dimension of the deepest layer of the unet

            flatten_latent_dim = query_states.shape[-3]
            if flatten_latent_dim % 64 == 0:
                query_chunk_size = int(flatten_latent_dim / 64)
            elif flatten_latent_dim % 16 == 0:
                query_chunk_size = int(flatten_latent_dim / 16)
            elif flatten_latent_dim % 4 == 0:
                query_chunk_size = int(flatten_latent_dim / 4)
            else:
                query_chunk_size = int(flatten_latent_dim)

            hidden_states = jax_memory_efficient_attention(
                query_states, key_states, value_states, query_chunk_size=query_chunk_size, key_chunk_size=4096 * 4
            )

            hidden_states = hidden_states.transpose(1, 0, 2)
        else:
            # compute attentions
            attention_scores = jnp.einsum("b i d, b j d->b i j", query_states, key_states)
            attention_scores = attention_scores * self.scale
            attention_probs = nn.softmax(attention_scores, axis=2)

            # attend to values
            hidden_states = jnp.einsum("b i j, b j d -> b i d", attention_probs, value_states)

        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        hidden_states = self.proj_attn(hidden_states)
        return hidden_states

class FlaxLoRALinearLayer(nn.Module):
    out_features: int
    dtype: jnp.dtype = jnp.float32
    rank: int=4

    def setup(self):
        self.down = nn.Dense(self.rank, use_bias=False, kernel_init=nn.initializers.normal(stddev=1 / self.rank), dtype=self.dtype)
        self.up = nn.Dense(self.out_features, use_bias=False, kernel_init=nn.initializers.zeros, dtype=self.dtype)

    def __call__(self, hidden_states):
        down_hidden_states = self.down(hidden_states)
        up_hidden_states = self.up(down_hidden_states)
        return up_hidden_states

class FlaxLoRACrossFrameAttention(nn.Module):
    r"""
    A Flax multi-head attention module as described in: https://arxiv.org/abs/1706.03762

    Parameters:
        query_dim (:obj:`int`):
            Input hidden states dimension
        heads (:obj:`int`, *optional*, defaults to 8):
            Number of heads
        dim_head (:obj:`int`, *optional*, defaults to 64):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        batch_size: The number that represents actual batch size, other than the frames.
            For example, using calling unet with a single prompt and num_images_per_prompt=1, batch_size should be
            equal to 2, due to classifier-free guidance.

    """
    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    use_memory_efficient_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    batch_size : int = 2
    rank: int=4

    def setup(self):
        inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head**-0.5

        # Weights were exported with old names {to_q, to_k, to_v, to_out}
        self.query = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_q")
        self.key = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_k")
        self.value = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_v")

        self.add_k_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype)
        self.add_v_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype)

        self.proj_attn = nn.Dense(self.query_dim, dtype=self.dtype, name="to_out_0")

        self.to_q_lora = FlaxLoRALinearLayer(inner_dim, rank=self.rank, dtype=self.dtype)
        self.to_k_lora = FlaxLoRALinearLayer(inner_dim, rank=self.rank, dtype=self.dtype)
        self.to_v_lora = FlaxLoRALinearLayer(inner_dim, rank=self.rank, dtype=self.dtype)
        self.to_out_lora = FlaxLoRALinearLayer(inner_dim, rank=self.rank, dtype=self.dtype)

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def __call__(self, hidden_states, context=None, deterministic=True, scale=1.):
        is_cross_attention = context is not None
        context = hidden_states if context is None else context
        query_proj = self.query(hidden_states) + scale * self.to_q_lora(hidden_states)
        key_proj = self.key(context) + scale * self.to_k_lora(hidden_states)
        value_proj = self.value(context) + scale * self.to_v_lora(hidden_states)

        # Sparse Attention
        if not is_cross_attention:
            video_length = 1 if key_proj.shape[0] < self.batch_size else key_proj.shape[0] // self.batch_size
            #first_frame_index = [0] * video_length
            #first frame ==> previous frame
            previous_frame_index = jnp.array([0] + list(range(video_length - 1)))

            # rearrange keys to have batch and frames in the 1st and 2nd dims respectively
            key_proj = rearrange_3(key_proj, video_length)
            key_proj = key_proj[:, previous_frame_index]
            # rearrange values to have batch and frames in the 1st and 2nd dims respectively
            value_proj = rearrange_3(value_proj, video_length)
            value_proj = value_proj[:, previous_frame_index]

            # rearrange back to original shape
            key_proj = rearrange_4(key_proj)
            value_proj = rearrange_4(value_proj)

        query_states = self.reshape_heads_to_batch_dim(query_proj)
        key_states = self.reshape_heads_to_batch_dim(key_proj)
        value_states = self.reshape_heads_to_batch_dim(value_proj)

        if self.use_memory_efficient_attention:
            query_states = query_states.transpose(1, 0, 2)
            key_states = key_states.transpose(1, 0, 2)
            value_states = value_states.transpose(1, 0, 2)

            # this if statement create a chunk size for each layer of the unet
            # the chunk size is equal to the query_length dimension of the deepest layer of the unet

            flatten_latent_dim = query_states.shape[-3]
            if flatten_latent_dim % 64 == 0:
                query_chunk_size = int(flatten_latent_dim / 64)
            elif flatten_latent_dim % 16 == 0:
                query_chunk_size = int(flatten_latent_dim / 16)
            elif flatten_latent_dim % 4 == 0:
                query_chunk_size = int(flatten_latent_dim / 4)
            else:
                query_chunk_size = int(flatten_latent_dim)

            hidden_states = jax_memory_efficient_attention(
                query_states, key_states, value_states, query_chunk_size=query_chunk_size, key_chunk_size=4096 * 4
            )

            hidden_states = hidden_states.transpose(1, 0, 2)
        else:
            # compute attentions
            attention_scores = jnp.einsum("b i d, b j d->b i j", query_states, key_states)
            attention_scores = attention_scores * self.scale
            attention_probs = nn.softmax(attention_scores, axis=2)

            # attend to values
            hidden_states = jnp.einsum("b i j, b j d -> b i d", attention_probs, value_states)

        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        hidden_states = self.proj_attn(hidden_states) + scale * self.to_out_lora(hidden_states)
        return hidden_states

class FlaxBasicTransformerBlock(nn.Module):
    r"""
    A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
    https://arxiv.org/abs/1706.03762


    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        only_cross_attention (`bool`, defaults to `False`):
            Whether to only apply cross attention.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
    """
    dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = False

    def setup(self):

        # self attention (or cross_attention if only_cross_attention is True)
        self.attn1 = FlaxCrossFrameAttention(
                                        self.dim, self.n_heads, self.d_head, self.dropout, self.use_memory_efficient_attention, dtype=self.dtype,
                                        )
        # cross attention
        self.attn2 = FlaxCrossFrameAttention(
                                        self.dim, self.n_heads, self.d_head, self.dropout, self.use_memory_efficient_attention, dtype=self.dtype,
                                        )
        self.ff = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
        self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)


    def __call__(self, hidden_states, context, deterministic=True):
        # self attention
        residual = hidden_states

        if self.only_cross_attention:
            hidden_states = self.attn1(self.norm1(hidden_states), context, deterministic=deterministic)
        else:
            hidden_states = self.attn1(self.norm1(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states

        hidden_states = self.attn2(self.norm2(hidden_states), context, deterministic=deterministic)

        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff(self.norm3(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        return hidden_states

class FlaxLoRABasicTransformerBlock(nn.Module):
    r"""
    A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
    https://arxiv.org/abs/1706.03762


    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        only_cross_attention (`bool`, defaults to `False`):
            Whether to only apply cross attention.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
    """
    dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = False

    def setup(self):

        # self attention (or cross_attention if only_cross_attention is True)
        self.attn1 = FlaxLoRACrossFrameAttention(
                                        self.dim, self.n_heads, self.d_head, self.dropout, self.use_memory_efficient_attention, dtype=self.dtype,
                                        )
        # cross attention
        self.attn2 = FlaxLoRACrossFrameAttention(
                                        self.dim, self.n_heads, self.d_head, self.dropout, self.use_memory_efficient_attention, dtype=self.dtype,
                                        )
        self.ff = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
        self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)


    def __call__(self, hidden_states, context, deterministic=True, scale=1.):
        # self attention
        residual = hidden_states

        if self.only_cross_attention:
            hidden_states = self.attn1(self.norm1(hidden_states), context, deterministic=deterministic, scale=scale)
        else:
            hidden_states = self.attn1(self.norm1(hidden_states), deterministic=deterministic, scale=scale)
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states

        hidden_states = self.attn2(self.norm2(hidden_states), context, deterministic=deterministic, scale=scale)

        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff(self.norm3(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + residual

        return hidden_states


class FlaxCrossFrameTransformer2DModel(nn.Module):
    r"""
    A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
    https://arxiv.org/pdf/1506.02025.pdf


    Parameters:
        in_channels (:obj:`int`):
            Input number of channels
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        depth (:obj:`int`, *optional*, defaults to 1):
            Number of transformers block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_linear_projection (`bool`, defaults to `False`): tbd
        only_cross_attention (`bool`, defaults to `False`): tbd
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
    """
    in_channels: int
    n_heads: int
    d_head: int
    depth: int = 1
    dropout: float = 0.0
    use_linear_projection: bool = False
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = False

    def setup(self):
        self.norm = nn.GroupNorm(num_groups=32, epsilon=1e-5)

        inner_dim = self.n_heads * self.d_head
        if self.use_linear_projection:
            self.proj_in = nn.Dense(inner_dim, dtype=self.dtype)
        else:
            self.proj_in = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

        self.transformer_blocks = [
            FlaxBasicTransformerBlock(
                inner_dim,
                self.n_heads,
                self.d_head,
                dropout=self.dropout,
                only_cross_attention=self.only_cross_attention,
                dtype=self.dtype,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
            )
            for _ in range(self.depth)
        ]

        if self.use_linear_projection:
            self.proj_out = nn.Dense(inner_dim, dtype=self.dtype)
        else:
            self.proj_out = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

    def __call__(self, hidden_states, context, deterministic=True):
        batch, height, width, channels = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height * width, channels)
            hidden_states = self.proj_in(hidden_states)
        else:
            hidden_states = self.proj_in(hidden_states)
            hidden_states = hidden_states.reshape(batch, height * width, channels)

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, context, deterministic=deterministic)

        if self.use_linear_projection:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, channels)
        else:
            hidden_states = hidden_states.reshape(batch, height, width, channels)
            hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states + residual
        return hidden_states

class FlaxLoRACrossFrameTransformer2DModel(nn.Module):
    r"""
    A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
    https://arxiv.org/pdf/1506.02025.pdf


    Parameters:
        in_channels (:obj:`int`):
            Input number of channels
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        depth (:obj:`int`, *optional*, defaults to 1):
            Number of transformers block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_linear_projection (`bool`, defaults to `False`): tbd
        only_cross_attention (`bool`, defaults to `False`): tbd
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
    """
    in_channels: int
    n_heads: int
    d_head: int
    depth: int = 1
    dropout: float = 0.0
    use_linear_projection: bool = False
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = False

    def setup(self):
        self.norm = nn.GroupNorm(num_groups=32, epsilon=1e-5)

        inner_dim = self.n_heads * self.d_head
        if self.use_linear_projection:
            self.proj_in = nn.Dense(inner_dim, dtype=self.dtype)
        else:
            self.proj_in = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

        self.transformer_blocks = [
            FlaxLoRABasicTransformerBlock(
                inner_dim,
                self.n_heads,
                self.d_head,
                dropout=self.dropout,
                only_cross_attention=self.only_cross_attention,
                dtype=self.dtype,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
            )
            for _ in range(self.depth)
        ]

        if self.use_linear_projection:
            self.proj_out = nn.Dense(inner_dim, dtype=self.dtype)
        else:
            self.proj_out = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

    def __call__(self, hidden_states, context, deterministic=True, scale=1.0):
        batch, height, width, channels = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height * width, channels)
            hidden_states = self.proj_in(hidden_states)
        else:
            hidden_states = self.proj_in(hidden_states)
            hidden_states = hidden_states.reshape(batch, height * width, channels)

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, context, deterministic=deterministic, scale=scale)

        if self.use_linear_projection:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, channels)
        else:
            hidden_states = hidden_states.reshape(batch, height, width, channels)
            hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states + residual
        return hidden_states

