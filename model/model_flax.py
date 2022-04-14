from typing import Optional, Tuple, Union, List

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from transformers.utils import ModelOutput

from config import FunnelAeConfig
from modeling_funnel_flax import (
    FlaxFunnelPreTrainedModel, FunnelAttentionStructure, FunnelLayer,
    dense_std, ACT2FN
)


@flax.struct.dataclass
class FlaxFunnelAeOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length / 2^n_blocks, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        latent_codes (`tuple(jnp.ndarray)` of shape `(batch_size, sequence_length / 2^nth_block, latent_size)`):
            Tuple of latents taken after each block.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: jnp.ndarray = None
    latent_codes: Tuple[jnp.ndarray] = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


class LatentFFN(nn.Module):
    config: FunnelAeConfig
    d_out: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        std = dense_std(self.config, self.config.d_inner + self.d_out)
        self.linear_1 = nn.Dense(self.config.d_inner, kernel_init=jax.nn.initializers.normal(std), dtype=self.dtype)
        self.activation_function = ACT2FN[self.config.hidden_act]
        self.activation_dropout = nn.Dropout(self.config.activation_dropout)
        self.linear_2 = nn.Dense(self.d_out, kernel_init=jax.nn.initializers.normal(std), dtype=self.dtype)
        self.layer_norm = nn.LayerNorm(self.config.layer_norm_eps)

    def __call__(self, hidden: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        h = self.linear_1(hidden)
        h = self.activation_function(h)
        h = self.activation_dropout(h, deterministic=deterministic)
        h = self.linear_2(h)
        return self.layer_norm(h)


class FlaxAe(nn.Module):
    config: FunnelAeConfig
    is_encoder: bool
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.is_encoder:
            self.d_enc_latent = self.config.encoder_d_latent
            self.d_latent = self.config.encoder_d_latent
        else:
            self.d_enc_latent = self.config.decoder_d_latent
            self.d_latent = self.config.decoder_d_latent + self.config.encoder_d_latent
        self.encoder = LatentFFN(self.config, self.d_enc_latent, dtype=self.dtype)
        self.decoder = LatentFFN(self.config, self.d_latent, dtype=self.dtype)

    def __call__(self, hidden: jnp.ndarray, extra_latents: jnp.ndarray = None, deterministic: bool = True) -> jnp.ndarray:
        latent = self.encoder(hidden, deterministic=deterministic)
        if self.is_encoder:
            # append encoder latents as extra dimensions on the same tokens
            h = self.decoder(jnp.concatenate(latent, extra_latents, axis=-1), deterministic=deterministic)
        else:
            h = self.decoder(latent, deterministic=deterministic)
        return self.layer_norm(hidden), latent


class FlaxFunnelBlock(nn.Module):
    config: FunnelAeConfig
    is_encoder: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention_structure = FunnelAttentionStructure(self.config)
        self.blocks = [
            [
                [
                    FunnelLayer(self.config, block_index, dtype=self.dtype)
                    for _ in range(block_size)
                ]
                for block_index, block_size in enumerate(self.config.block_sizes)
            ]
        ]
        self.latent_ffn = [
            FlaxAe(self.config, self.is_encoder, dtype=self.dtype)
            for _ in self.config.block_sizes
        ]

    def encode(
        self,
        inputs_embeds: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        token_type_ids: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # The pooling is not implemented on long tensors, so we convert this mask.
        attention_mask = attention_mask.type_as(inputs_embeds)
        attention_inputs = self.attention_structure.init_attention_inputs(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = inputs_embeds

        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        latent_codes = ()

        for block_index, block in enumerate(self.blocks):
            pooling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )
            for (layer_index, layer) in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden if self.config.pool_q_only else pooled_hidden
                    else:
                        query = key = value = hidden
                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions, deterministic=deterministic)
                    hidden = layer_output[0]
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)
            hidden, latent = self.latent_ffn[block_index](hidden, deterministic=deterministic)
            all_hidden_states = all_hidden_states + (hidden,)
            latent_codes = latent_codes + (latent,)

        if not return_dict:
            return tuple(v for v in [hidden, latent_codes, all_hidden_states, all_attentions] if v is not None)
        return FlaxFunnelAeOutput(last_hidden_state=hidden, latent_codes=latent_codes, hidden_states=all_hidden_states, attentions=all_attentions)

    def decode(
        self,
        hidden: jnp.ndarray,
        encoder_latent_codes: Tuple[jnp.ndarray],
        attention_mask: Optional[jnp.ndarray] = None,
        token_type_ids: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        # TODO:
        # - Take encoder latents
        # - Combine with decoder latents to make new hidden states
        # - 
        #
        # The pooling is not implemented on long tensors, so we convert this mask.
        attention_mask = attention_mask.type_as(hidden)
        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        all_hidden_states = (hidden,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        latent_codes = ()

        for block_index, block in enumerate(self.blocks):
            pooling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )
            for (layer_index, layer) in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden if self.config.pool_q_only else pooled_hidden
                    else:
                        query = key = value = hidden
                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions, deterministic=deterministic)
                    hidden = layer_output[0]
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)
            hidden, latent = self.latent_ffn[block_index](hidden, encoder_latent_codes[-1-block_index], deterministic=deterministic)
            all_hidden_states = all_hidden_states + (hidden,)
            latent_codes = latent_codes + (latent,)

        if not return_dict:
            return tuple(v for v in [hidden, latent_codes, all_hidden_states, all_attentions] if v is not None)
        return FlaxFunnelAeOutput(last_hidden_state=hidden, latent_codes=latent_codes, hidden_states=all_hidden_states, attentions=all_attentions)


class FlaxFunnelAeModule(nn.Module):
    config: FunnelAeConfig
    dtype: jnp.dtype = jnp.float32


class FlaxFunnelAeModel(FlaxFunnelPreTrainedModel):
    config: FunnelAeConfig
    dtype: jnp.dtype = jnp.float32
    module_class = FlaxFunnelAeModule

    def setup(self):
        self.encoder = FlaxFunnelBlock(config=self.config, dtype=self.dtype)
        self.decoder = self.encoder if self.config.share_encoder_blocks else FlaxFunnelBlock(config=self.config, dtype=self.dtype)
        if self.config.share_encoder_blocks:
            assert self.config.encoder_latent == self.config.decoder_latent

    def _get_encoder_module(self):
        return self.model.encoder

    def _get_decoder_module(self):
        return self.model.decoder

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        deterministic=True,
    ):
        encoder_outputs = self.encoder.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            deterministic=deterministic
        )
        encoder_outputs = self.decoder.decode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            deterministic=deterministic
        )
