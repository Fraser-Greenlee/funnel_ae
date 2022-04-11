from typing import Optional, Tuple, Union, List

import jax
import jax.numpy as jnp
import flax.linen as nn

from transformers.models.funnel.configuration_funnel import FunnelConfig


class FunnelEmbeddings(nn.Module):
    config: FunnelConfig

    def setup(self):
        # TODO account for missing `padding_idx`
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.layer_norm = nn.LayerNorm(self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.hidden_dropout)

    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = self.layer_norm(inputs_embeds)
        embeddings = self.dropout(embeddings, deterministic=deterministic)
        return embeddings


class FunnelAttentionStructure(nn.Module):
    """
    Contains helpers for `FunnelRelMultiheadAttention `.
    """

    cls_token_type_id: int = 2

    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        self.config = config
        self.sin_dropout = nn.Dropout(config.hidden_dropout)
        self.cos_dropout = nn.Dropout(config.hidden_dropout)
        # Track where we are at in terms of pooling from the original input, e.g., by how much the sequence length was
        # divided.
        self.pooling_mult = None

    def init_attention_inputs(
        self,
        inputs_embeds: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        token_type_ids: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray]:
        """Returns the attention inputs associated to the inputs of the model."""
        # inputs_embeds has shape batch_size x seq_len x d_model
        # attention_mask and token_type_ids have shape batch_size x seq_len
        self.pooling_mult = 1
        self.seq_len = seq_len = inputs_embeds.size(1)
        position_embeds = self.get_position_embeds(seq_len, inputs_embeds.dtype, inputs_embeds.device)
        token_type_mat = self.token_type_ids_to_mat(token_type_ids) if token_type_ids is not None else None
        # TODO does this work?
        cls_mask = (
            nn.functional.pad(inputs_embeds.new_ones([seq_len - 1, seq_len - 1]), (1, 0, 1, 0))
            if self.config.separate_cls
            else None
        )
        return (position_embeds, token_type_mat, attention_mask, cls_mask)

    def token_type_ids_to_mat(self, token_type_ids: jnp.ndarray) -> jnp.ndarray:
        """Convert `token_type_ids` to `token_type_mat`."""
        token_type_mat = token_type_ids[:, :, None] == token_type_ids[:, None]
        # Treat <cls> as in the same segment as both A & B
        cls_ids = token_type_ids == self.cls_token_type_id
        cls_mat = cls_ids[:, :, None] | cls_ids[:, None]
        return cls_mat | token_type_mat

    def get_position_embeds(
        self, seq_len: int, dtype: jnp.dtype
    ) -> Union[Tuple[jnp.ndarray], List[List[jnp.ndarray]]]:
        """
        Create and cache inputs related to relative position encoding. Those are very different depending on whether we
        are using the factorized or the relative shift attention:

        For the factorized attention, it returns the matrices (phi, pi, psi, omega) used in the paper, appendix A.2.2,
        final formula.

        For the relative shift attention, it returns all possible vectors R used in the paper, appendix A.2.1, final
        formula.

        Paper link: https://arxiv.org/abs/2006.03236
        """
        d_model = self.config.d_model
        if self.config.attention_type == "factorized":
            # Notations from the paper, appending A.2.2, final formula.
            # We need to create and return the matrices phi, psi, pi and omega.
            pos_seq = jnp.arange(0, seq_len, 1.0, dtype=dtype)
            freq_seq = jnp.arange(0, d_model // 2, 1.0, dtype=dtype)
            inv_freq = 1 / (10000 ** (freq_seq / (d_model // 2)))
            sinusoid = pos_seq[:, None] * inv_freq[None]
            sin_embed = jnp.sin(sinusoid)
            sin_embed_d = self.sin_dropout(sin_embed)
            cos_embed = jnp.cos(sinusoid)
            cos_embed_d = self.cos_dropout(cos_embed)
            # This is different from the formula on the paper...
            phi = jnp.concatenate([sin_embed_d, sin_embed_d], dim=-1)
            psi = jnp.concatenate([cos_embed, sin_embed], dim=-1)
            pi = jnp.concatenate([cos_embed_d, cos_embed_d], dim=-1)
            omega = jnp.concatenate([-sin_embed, cos_embed], dim=-1)
            return (phi, pi, psi, omega)
        else:
            # Notations from the paper, appending A.2.1, final formula.
            # We need to create and return all the possible vectors R for all blocks and shifts.
            freq_seq = jnp.arange(0, d_model // 2, 1.0, dtype=dtype)
            inv_freq = 1 / (10000 ** (freq_seq / (d_model // 2)))
            # Maximum relative positions for the first input
            rel_pos_id = jnp.arange(-seq_len * 2, seq_len * 2, 1.0, dtype=dtype)
            zero_offset = seq_len * 2
            sinusoid = rel_pos_id[:, None] * inv_freq[None]
            sin_embed = self.sin_dropout(jnp.sin(sinusoid))
            cos_embed = self.cos_dropout(jnp.cos(sinusoid))
            pos_embed = jnp.concatenate([sin_embed, cos_embed], dim=-1)

            pos = jnp.arange(0, seq_len, dtype=dtype)
            pooled_pos = pos
            position_embeds_list = []
            for block_index in range(0, self.config.num_blocks):
                # For each block with block_index > 0, we need two types position embeddings:
                #   - Attention(pooled-q, unpooled-kv)
                #   - Attention(pooled-q, pooled-kv)
                # For block_index = 0 we only need the second one and leave the first one as None.

                # First type
                if block_index == 0:
                    position_embeds_pooling = None
                else:
                    pooled_pos = self.stride_pool_pos(pos, block_index)

                    # construct rel_pos_id
                    stride = 2 ** (block_index - 1)
                    rel_pos = self.relative_pos(pos, stride, pooled_pos, shift=2)
                    rel_pos = rel_pos[:, None] + zero_offset
                    rel_pos = rel_pos.expand(rel_pos.size(0), d_model)
                    # TODO try jax.lax.gather?
                    position_embeds_pooling = torch.gather(pos_embed, 0, rel_pos)

                # Second type
                pos = pooled_pos
                stride = 2**block_index
                rel_pos = self.relative_pos(pos, stride)

                rel_pos = rel_pos[:, None] + zero_offset
                rel_pos = rel_pos.expand(rel_pos.size(0), d_model)
                # TODO try jax.lax.gather?
                position_embeds_no_pooling = torch.gather(pos_embed, 0, rel_pos)

                position_embeds_list.append([position_embeds_no_pooling, position_embeds_pooling])
            return position_embeds_list

    def stride_pool_pos(self, pos_id: jnp.ndarray, block_index: int):
        """
        Pool `pos_id` while keeping the cls token separate (if `config.separate_cls=True`).
        """
        if self.config.separate_cls:
            # Under separate <cls>, we treat the <cls> as the first token in
            # the previous block of the 1st real block. Since the 1st real
            # block always has position 1, the position of the previous block
            # will be at `1 - 2 ** block_index`.
            cls_pos = pos_id.new_tensor([-(2**block_index) + 1])
            pooled_pos_id = pos_id[1:-1] if self.config.truncate_seq else pos_id[1:]
            return jnp.concatenate([cls_pos, pooled_pos_id[::2]], 0)
        else:
            return pos_id[::2]

    def relative_pos(self, pos: jnp.ndarray, stride: int, pooled_pos=None, shift: int = 1) -> jnp.ndarray:
        """
        Build the relative positional vector between `pos` and `pooled_pos`.
        """
        if pooled_pos is None:
            pooled_pos = pos

        ref_point = pooled_pos[0] - pos[0]
        num_remove = shift * len(pooled_pos)
        max_dist = ref_point + num_remove * stride
        min_dist = pooled_pos[0] - pos[-1]

        return jnp.arange(max_dist, min_dist - 1, -stride, dtype=jnp.int64, device=pos.device)

    def stride_pool(
        self,
        tensor: Union[jnp.ndarray, Tuple[jnp.ndarray], List[jnp.ndarray]],
        axis: Union[int, Tuple[int], List[int]],
    ) -> jnp.ndarray:
        """
        Perform pooling by stride slicing the tensor along the given axis.
        """
        if tensor is None:
            return None

        # Do the stride pool recursively if axis is a list or a tuple of ints.
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                tensor = self.stride_pool(tensor, ax)
            return tensor

        # Do the stride pool recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.stride_pool(x, axis) for x in tensor)

        # Deal with negative axis
        axis %= tensor.ndim

        axis_slice = (
            slice(None, -1, 2) if self.config.separate_cls and self.config.truncate_seq else slice(None, None, 2)
        )
        enc_slice = [slice(None)] * axis + [axis_slice]
        if self.config.separate_cls:
            cls_slice = [slice(None)] * axis + [slice(None, 1)]
            tensor = jnp.concatenate([tensor[cls_slice], tensor], axis=axis)
        return tensor[enc_slice]

    def pool_tensor(
        self, tensor: Union[jnp.ndarray, Tuple[jnp.ndarray], List[jnp.ndarray]], mode: str = "mean", stride: int = 2
    ) -> jnp.ndarray:
        """Apply 1D pooling to a tensor of size [B x T (x H)]."""
        if tensor is None:
            return None

        # Do the pool recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.pool_tensor(tensor, mode=mode, stride=stride) for x in tensor)

        if self.config.separate_cls:
            suffix = tensor[:, :-1] if self.config.truncate_seq else tensor
            tensor = jnp.concatenate([tensor[:, :1], suffix], dim=1)

        ndim = tensor.ndim
        if ndim == 2:
            tensor = tensor[:, None, :, None]
        elif ndim == 3:
            tensor = tensor[:, None, :, :]
        # Stride is applied on the second-to-last dimension.
        stride = (stride, 1)

        if mode == "mean":
            tensor = nn.functional.avg_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "max":
            tensor = nn.functional.max_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "min":
            tensor = -nn.functional.max_pool2d(-tensor, stride, stride=stride, ceil_mode=True)
        else:
            raise NotImplementedError("The supported modes are 'mean', 'max' and 'min'.")

        if ndim == 2:
            return tensor[:, 0, :, 0]
        elif ndim == 3:
            return tensor[:, 0]
        return tensor

    def pre_attention_pooling(
        self, output, attention_inputs: Tuple[jnp.ndarray]
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
        """Pool `output` and the proper parts of `attention_inputs` before the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config.pool_q_only:
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds[:2], 0) + position_embeds[2:]
            token_type_mat = self.stride_pool(token_type_mat, 1)
            cls_mask = self.stride_pool(cls_mask, 0)
            output = self.pool_tensor(output, mode=self.config.pooling_type)
        else:
            self.pooling_mult *= 2
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds, 0)
            token_type_mat = self.stride_pool(token_type_mat, [1, 2])
            cls_mask = self.stride_pool(cls_mask, [1, 2])
            attention_mask = self.pool_tensor(attention_mask, mode="min")
            output = self.pool_tensor(output, mode=self.config.pooling_type)
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return output, attention_inputs

    def post_attention_pooling(self, attention_inputs: Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray]:
        """Pool the proper parts of `attention_inputs` after the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config.pool_q_only:
            self.pooling_mult *= 2
            if self.config.attention_type == "factorized":
                position_embeds = position_embeds[:2] + self.stride_pool(position_embeds[2:], 0)
            token_type_mat = self.stride_pool(token_type_mat, 2)
            cls_mask = self.stride_pool(cls_mask, 1)
            attention_mask = self.pool_tensor(attention_mask, mode="min")
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return attention_inputs
