from typing import Optional, Tuple, Union, List

import jax
import jax.numpy as jnp
import flax.linen as nn

from transformers.models.funnel.configuration_funnel import FunnelConfig
from transformers.modeling_flax_utils import ACT2FN


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
    config: FunnelConfig
    cls_token_type_id: int = 2

    def setup(self):
        self.sin_dropout = nn.Dropout(self.config.hidden_dropout)
        self.cos_dropout = nn.Dropout(self.config.hidden_dropout)

    def init_attention_inputs(
        self,
        inputs_embeds: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        token_type_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        """Returns the attention inputs associated to the inputs of the model."""
        # inputs_embeds has shape batch_size x seq_len x d_model
        # attention_mask and token_type_ids have shape batch_size x seq_len
        seq_len = inputs_embeds.shape[1]
        position_embeds = self.get_position_embeds(seq_len, inputs_embeds.dtype, deterministic)
        token_type_mat = self.token_type_ids_to_mat(token_type_ids) if token_type_ids is not None else None
        cls_mask = (
            jnp.pad(jnp.ones([seq_len - 1, seq_len - 1], dtype=inputs_embeds.dtype), (1, 0))
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
        self, seq_len: int, dtype: jnp.dtype, deterministic: bool = True
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
            sin_embed_d = self.sin_dropout(sin_embed, deterministic=deterministic)
            cos_embed = jnp.cos(sinusoid)
            cos_embed_d = self.cos_dropout(cos_embed, deterministic=deterministic)
            # This is different from the formula on the paper...
            phi =   jnp.concatenate([sin_embed_d, sin_embed_d], axis=-1)
            psi =   jnp.concatenate([cos_embed, sin_embed],     axis=-1)
            pi =    jnp.concatenate([cos_embed_d, cos_embed_d], axis=-1)
            omega = jnp.concatenate([-sin_embed, cos_embed],    axis=-1)
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
            sin_embed = self.sin_dropout(jnp.sin(sinusoid), deterministic=deterministic)
            cos_embed = self.cos_dropout(jnp.cos(sinusoid), deterministic=deterministic)
            pos_embed = jnp.concatenate([sin_embed, cos_embed], axis=-1)

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
                    rel_pos = jnp.broadcast_to(rel_pos, (rel_pos.shape[0], d_model))
                    position_embeds_pooling = jnp.take_along_axis(pos_embed, rel_pos, 0)

                # Second type
                pos = pooled_pos
                stride = 2**block_index
                rel_pos = self.relative_pos(pos, stride)
                rel_pos = rel_pos[:, None] + zero_offset
                rel_pos = jnp.broadcast_to(rel_pos, (rel_pos.shape[0], d_model))
                position_embeds_no_pooling = jnp.take_along_axis(pos_embed, rel_pos, 0)

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
            cls_pos = jnp.array([-(2**block_index) + 1], dtype=pos_id.dtype)
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

        return jnp.arange(max_dist, min_dist - 1, -stride, dtype=jnp.int64)

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
        enc_slice = tuple([slice(None)] * axis + [axis_slice])
        if self.config.separate_cls:
            cls_slice = tuple([slice(None)] * axis + [slice(None, 1)])
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
            tensor = jnp.concatenate([tensor[:, :1], suffix], axis=1)

        ndim = tensor.ndim
        if ndim == 2:
            tensor = tensor[:, None, :, None]
        elif ndim == 3:
            tensor = tensor[:, None, :, :]

        # Stride is applied on the second-to-last dimension.
        stride = tuple([1 for _ in range(len(tensor.shape)-2)] + [stride])

        if mode == "mean":
            tensor = nn.avg_pool(tensor, stride, strides=stride, padding='SAME')
        elif mode == "max":
            tensor = nn.max_pool(tensor, stride, strides=stride, padding='SAME')
        elif mode == "min":
            tensor = -nn.max_pool(-tensor, stride, strides=stride, padding='SAME')
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
            if self.config.attention_type == "factorized":
                position_embeds = position_embeds[:2] + self.stride_pool(position_embeds[2:], 0)
            token_type_mat = self.stride_pool(token_type_mat, 2)
            cls_mask = self.stride_pool(cls_mask, 1)
            attention_mask = self.pool_tensor(attention_mask, mode="min")
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return attention_inputs


def _relative_shift_gather(positional_attn: jnp.ndarray, context_len: int, shift: int) -> jnp.ndarray:
    batch_size, n_head, seq_len, max_rel_len = positional_attn.shape
    # max_rel_len = 2 * context_len + shift -1 is the numbers of possible relative positions i-j

    # What's next is the same as doing the following gather, which might be clearer code but less efficient.
    # idxs = context_len + torch.arange(0, context_len).unsqueeze(0) - torch.arange(0, seq_len).unsqueeze(1)
    # # matrix of context_len + i-j
    # return positional_attn.gather(3, idxs.expand([batch_size, n_head, context_len, context_len]))

    positional_attn = jnp.reshape(positional_attn, [batch_size, n_head, max_rel_len, seq_len])
    positional_attn = positional_attn[:, :, shift:, :]
    positional_attn = jnp.reshape(positional_attn, [batch_size, n_head, seq_len, max_rel_len - shift])
    positional_attn = positional_attn[..., :context_len]
    return positional_attn


class FunnelRelMultiheadAttention(nn.Module):
    config: FunnelConfig
    block_index: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        d_model, n_head, d_head = self.config.d_model, self.config.n_head, self.config.d_head

        self.hidden_dropout = nn.Dropout(self.config.hidden_dropout)
        self.attention_dropout = nn.Dropout(self.config.attention_dropout)

        self.q_head = nn.Dense(n_head * d_head, use_bias=False, dtype=self.dtype)
        self.k_head = nn.Dense(n_head * d_head, dtype=self.dtype)
        self.v_head = nn.Dense(n_head * d_head, dtype=self.dtype)

        self.r_w_bias =  self.param('r_w_bias',  nn.initializers.zeros, [n_head,  d_head])
        self.r_r_bias =  self.param('r_r_bias',  nn.initializers.zeros, [n_head,  d_head])
        self.r_kernel =  self.param('r_kernel',  nn.initializers.zeros, [d_model, n_head, d_head])
        self.r_s_bias =  self.param('r_s_bias',  nn.initializers.zeros, [n_head,  d_head])
        self.seg_embed = self.param('seg_embed', nn.initializers.zeros, [2,       n_head, d_head])

        self.post_proj = nn.Dense(d_model, dtype=self.dtype)
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.scale = 1.0 / (d_head**0.5)

    def relative_positional_attention(self, position_embeds, q_head, context_len, cls_mask=None):
        """Relative attention score for the positional encodings"""
        # q_head has shape batch_size x sea_len x n_head x d_head
        if self.config.attention_type == "factorized":
            # Notations from the paper, appending A.2.2, final formula (https://arxiv.org/abs/2006.03236)
            # phi and pi have shape seq_len x d_model, psi and omega have shape context_len x d_model
            phi, pi, psi, omega = position_embeds
            # Shape n_head x d_head
            u = self.r_r_bias * self.scale
            # Shape d_model x n_head x d_head
            w_r = self.r_kernel

            # Shape batch_size x sea_len x n_head x d_model
            q_r_attention = jnp.einsum("binh,dnh->bind", q_head + u, w_r)
            q_r_attention_1 = q_r_attention * phi[:, None]
            q_r_attention_2 = q_r_attention * pi[:, None]

            # Shape batch_size x n_head x seq_len x context_len
            positional_attn = jnp.einsum("bind,jd->bnij", q_r_attention_1, psi) + jnp.einsum(
                "bind,jd->bnij", q_r_attention_2, omega
            )
        else:
            shift = 2 if q_head.shape[1] != context_len else 1
            # Notations from the paper, appending A.2.1, final formula (https://arxiv.org/abs/2006.03236)
            # Grab the proper positional encoding, shape max_rel_len x d_model
            r = position_embeds[self.block_index][shift - 1]
            # Shape n_head x d_head
            v = self.r_r_bias * self.scale
            # Shape d_model x n_head x d_head
            w_r = self.r_kernel

            # Shape max_rel_len x n_head x d_model
            r_head = jnp.einsum("td,dnh->tnh", r, w_r)
            # Shape batch_size x n_head x seq_len x max_rel_len
            positional_attn = jnp.einsum("binh,tnh->bnit", q_head + v, r_head)
            # Shape batch_size x n_head x seq_len x context_len
            positional_attn = _relative_shift_gather(positional_attn, context_len, shift)

        if cls_mask is not None:
            positional_attn *= cls_mask
        return positional_attn

    def relative_token_type_attention(self, token_type_mat, q_head, cls_mask=None):
        """Relative attention score for the token_type_ids"""
        if token_type_mat is None:
            return 0
        batch_size, seq_len, context_len = token_type_mat.shape
        # q_head has shape batch_size x seq_len x n_head x d_head
        # Shape n_head x d_head
        r_s_bias = self.r_s_bias * self.scale

        # Shape batch_size x n_head x seq_len x 2
        token_type_bias = jnp.einsum("bind,snd->bnis", q_head + r_s_bias, self.seg_embed)
        # Shape batch_size x n_head x seq_len x context_len
        token_type_mat = jnp.broadcast_to(token_type_mat[:, None], (batch_size, q_head.shape[2], seq_len, context_len))
        # Shapes batch_size x n_head x seq_len
        diff_token_type, same_token_type = jnp.split(token_type_bias, 2, axis=-1)
        # Shape batch_size x n_head x seq_len x context_len
        token_type_attn = jnp.where(
            token_type_mat, jnp.broadcast_to(same_token_type, token_type_mat.shape), jnp.broadcast_to(diff_token_type, token_type_mat.shape)
        )

        if cls_mask is not None:
            token_type_attn *= cls_mask
        return token_type_attn

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        attention_inputs: Tuple[jnp.ndarray],
        output_attentions: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, ...]:
        # query has shape batch_size x seq_len x d_model
        # key and value have shapes batch_size x context_len x d_model
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs

        batch_size, seq_len, _ = query.shape
        context_len = key.shape[1]
        n_head, d_head = self.config.n_head, self.config.d_head

        # Shape batch_size x seq_len x n_head x d_head
        q_head = self.q_head(query).reshape(batch_size, seq_len, n_head, d_head)
        # Shapes batch_size x context_len x n_head x d_head
        k_head = self.k_head(key).reshape(batch_size, context_len, n_head, d_head)
        v_head = self.v_head(value).reshape(batch_size, context_len, n_head, d_head)

        q_head = q_head * self.scale
        # Shape n_head x d_head
        r_w_bias = self.r_w_bias * self.scale
        # Shapes batch_size x n_head x seq_len x context_len
        content_score = jnp.einsum("bind,bjnd->bnij", q_head + r_w_bias, k_head)
        positional_attn = self.relative_positional_attention(position_embeds, q_head, context_len, cls_mask)
        token_type_attn = self.relative_token_type_attention(token_type_mat, q_head, cls_mask)

        # merge attention scores
        attn_score = content_score + positional_attn + token_type_attn
        # TODO should I include old `precision safe in case of mixed precision training`?

        # perform masking
        if attention_mask is not None:
            attn_score = attn_score - 1e6 * (1 - attention_mask[:, None, None].astype(self.dtype))
        # attention probability
        attn_prob = jax.nn.softmax(attn_score, axis=-1)
        attn_prob = self.attention_dropout(attn_prob, deterministic=deterministic)

        # attention output, shape batch_size x seq_len x n_head x d_head
        attn_vec = jnp.einsum("bnij,bjnd->bind", attn_prob, v_head)

        # Shape shape batch_size x seq_len x d_model
        attn_out = self.post_proj(attn_vec.reshape(batch_size, seq_len, n_head * d_head))
        attn_out = self.hidden_dropout(attn_out, deterministic=deterministic)

        output = self.layer_norm(query + attn_out)
        return (output, attn_prob) if output_attentions else (output,)


class FunnelPositionwiseFFN(nn.Module):
    config: FunnelConfig

    def setup(self):
        self.linear_1 = nn.Dense(self.config.d_inner)
        self.activation_function = ACT2FN[self.config.hidden_act]
        self.activation_dropout = nn.Dropout(self.config.activation_dropout)
        self.linear_2 = nn.Dense(self.config.d_model)
        self.dropout = nn.Dropout(self.config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(self.config.layer_norm_eps)

    def __call__(self, hidden: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        h = self.linear_1(hidden)
        h = self.activation_function(h)
        h = self.activation_dropout(h, deterministic=deterministic)
        h = self.linear_2(h)
        h = self.dropout(h, deterministic=deterministic)
        return self.layer_norm(hidden + h)

