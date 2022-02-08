import numpy as np
import torch
from typing import Optional
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.utils import logging
from transformers.models.funnel.modeling_funnel import (
    FunnelEncoder, FunnelAttentionStructure, FunnelLayer, upsample
)
from transformers.modeling_outputs import BaseModelOutput
from transformers import (
    FunnelPreTrainedModel,
    PretrainedConfig,
)

from model.config import FunnelAeConfig
from model.outputs import AutoEncOutput


logger = logging.get_logger(__name__)


class FunnelAttentionUpsampleStructure(FunnelAttentionStructure):
    def __init__(self, config):
        super().__init__(config)
        # Track where we are at in terms of upsampling from the original input, e.g., by how much the sequence length was
        # divided.
        del self.upsampling_mult
        self.upsampling_mult = None

    def get_position_embeds(self, seq_len, dtype, device):
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
            pos_seq = torch.arange(0, seq_len, 1.0, dtype=dtype, device=device)
            freq_seq = torch.arange(0, d_model // 2, 1.0, dtype=dtype, device=device)
            inv_freq = 1 / (10000 ** (freq_seq / (d_model // 2)))
            sinusoid = pos_seq[:, None] * inv_freq[None]
            sin_embed = torch.sin(sinusoid)
            sin_embed_d = self.sin_dropout(sin_embed)
            cos_embed = torch.cos(sinusoid)
            cos_embed_d = self.cos_dropout(cos_embed)
            # This is different from the formula on the paper...
            phi = torch.cat([sin_embed_d, sin_embed_d], dim=-1)
            psi = torch.cat([cos_embed, sin_embed], dim=-1)
            pi = torch.cat([cos_embed_d, cos_embed_d], dim=-1)
            omega = torch.cat([-sin_embed, cos_embed], dim=-1)
            return (phi, pi, psi, omega)
        else:
            # Notations from the paper, appending A.2.1, final formula.
            # We need to create and return all the possible vectors R for all blocks and shifts.
            freq_seq = torch.arange(0, d_model // 2, 1.0, dtype=dtype, device=device)
            inv_freq = 1 / (10000 ** (freq_seq / (d_model // 2)))
            # Maximum relative positions for the first input
            rel_pos_id = torch.arange(-seq_len * 2, seq_len * 2, 1.0, dtype=dtype, device=device)
            zero_offset = seq_len * 2
            sinusoid = rel_pos_id[:, None] * inv_freq[None]
            sin_embed = self.sin_dropout(torch.sin(sinusoid))
            cos_embed = self.cos_dropout(torch.cos(sinusoid))
            pos_embed = torch.cat([sin_embed, cos_embed], dim=-1)

            pos = torch.arange(0, seq_len, dtype=dtype, device=device)
            upsampled_pos = pos
            position_embeds_list = []
            for block_index in range(0, self.config.num_blocks):
                # For each block with block_index > 0, we need two types position embeddings:
                #   - Attention(upsampled-q, unupsampled-kv)
                #   - Attention(upsampled-q, upsampled-kv)
                # For block_index = 0 we only need the second one and leave the first one as None.

                # First type
                if block_index == 0:
                    position_embeds_upsampling = None
                else:
                    upsampled_pos = self.stride_upsample_pos(pos, block_index)

                    # construct rel_pos_id
                    stride = 2 ** (block_index - 1)
                    rel_pos = self.relative_pos(pos, stride, upsampled_pos, shift=2)
                    rel_pos = rel_pos[:, None] + zero_offset
                    rel_pos = rel_pos.expand(rel_pos.size(0), d_model)
                    position_embeds_upsampling = torch.gather(pos_embed, 0, rel_pos)

                # Second type
                pos = upsampled_pos
                stride = 2 ** block_index
                rel_pos = self.relative_pos(pos, stride)

                rel_pos = rel_pos[:, None] + zero_offset
                rel_pos = rel_pos.expand(rel_pos.size(0), d_model)
                position_embeds_no_upsampling = torch.gather(pos_embed, 0, rel_pos)

                position_embeds_list.append([position_embeds_no_upsampling, position_embeds_upsampling])
            return position_embeds_list

    def stride_upsample_pos(self, pos_id, block_index):
        """
        Upsample `pos_id` while keeping the cls token separate (if `config.separate_cls=True`).
        """
        # TODO
        if self.config.separate_cls:
            # Under separate <cls>, we treat the <cls> as the first token in
            # the previous block of the 1st real block. Since the 1st real
            # block always has position 1, the position of the previous block
            # will be at `1 - 2 ** block_index`.
            cls_pos = pos_id.new_tensor([-(2 ** block_index) + 1])
            upsampled_pos_id = pos_id[1:-1] if self.config.truncate_seq else pos_id[1:]
            return torch.cat([cls_pos, upsampled_pos_id[::2]], 0)
        else:
            return pos_id[::2]

    def relative_pos(self, pos, stride, upsampled_pos=None, shift=1):
        """
        Build the relative positional vector between `pos` and `upsampled_pos`.
        """
        # TODO
        if upsampled_pos is None:
            upsampled_pos = pos

        ref_point = upsampled_pos[0] - pos[0]
        num_remove = shift * len(upsampled_pos)
        max_dist = ref_point + num_remove * stride
        min_dist = upsampled_pos[0] - pos[-1]

        return torch.arange(max_dist, min_dist - 1, -stride, dtype=torch.long, device=pos.device)

    def stride_upsample(self, tensor, axis):
        """
        Perform upsample by stride slicing the tensor along the given axis.
        """
        if tensor is None:
            return None

        # Do the stride upsample recursively if axis is a list or a tuple of ints.
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                tensor = self.stride_upsample(tensor, ax)
            return tensor

        # Do the stride upsample recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.stride_upsample(x, axis) for x in tensor)

        # Deal with negative axis
        axis %= tensor.ndim

        axis_slice = (
            slice(None, -1, 2) if self.config.separate_cls and self.config.truncate_seq else slice(None, None, 2)
        )
        enc_slice = [slice(None)] * axis + [axis_slice]
        if self.config.separate_cls:
            cls_slice = [slice(None)] * axis + [slice(None, 1)]
            tensor = torch.cat([tensor[cls_slice], tensor], axis=axis)

        # TODO switch to pooling
        breakpoint()

        return tensor[enc_slice]

    def upsample_tensor(self, tensor, mode="bilinear"):
        """Apply 1D upsample to a tensor of size [B x T (x H)]."""
        # TODO
        if tensor is None:
            return None

        # Do the upsample recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.upsample_tensor(tensor, mode=mode) for x in tensor)

        if self.config.separate_cls:
            suffix = tensor[:, :-1] if self.config.truncate_seq else tensor
            tensor = torch.cat([tensor[:, :1], suffix], dim=1)

        ndim = tensor.ndim
        if ndim == 2:
            tensor = tensor[:, None, :, None]
        elif ndim == 3:
            tensor = tensor[:, None, :, :]

        if mode == "nearest":
            tensor = nn.Upsample(scale_factor=2, mode='nearest')(tensor)
        elif mode == "bilinear":
            tensor = nn.Upsample(scale_factor=2, mode='bilinear')(tensor)
        elif mode == "bilinear_align_corners":
            tensor = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(tensor)
        else:
            raise NotImplementedError("The supported modes are 'nearest', 'bilinear' and 'bilinear_align_corners'.")

        # TODO what is this doing?

        if ndim == 2:
            return tensor[:, 0, :, 0]
        elif ndim == 3:
            return tensor[:, 0]
        return tensor

    def pre_attention_upsampling(self, output, attention_inputs):
        """Upsample `output` and the proper parts of `attention_inputs` before the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config.upsample_q_only:
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_upsample(position_embeds[:2], 0) + position_embeds[2:]
            token_type_mat = self.stride_upsample(token_type_mat, 1)
            cls_mask = self.stride_upsample(cls_mask, 0)
            output = self.upsample_tensor(output, mode=self.config.upsampling_type)
        else:
            self.upsampling_mult *= 2
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_upsample(position_embeds, 0)
            token_type_mat = self.stride_upsample(token_type_mat, [1, 2])
            cls_mask = self.stride_upsample(cls_mask, [1, 2])
            attention_mask = self.upsample_tensor(attention_mask, mode="nearest")
            output = self.upsample_tensor(output, mode=self.config.upsampling_type)
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return output, attention_inputs

    def post_attention_upsampling(self, attention_inputs):
        """Upsample the proper parts of `attention_inputs` after the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config.upsample_q_only:
            self.upsampling_mult *= 2
            if self.config.attention_type == "factorized":
                position_embeds = position_embeds[:2] + self.stride_upsample(position_embeds[2:], 0)
            token_type_mat = self.stride_upsample(token_type_mat, 2)
            cls_mask = self.stride_upsample(cls_mask, 1)
            attention_mask = self.upsample_tensor(attention_mask, mode="nearest")
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return attention_inputs


class FunnelAeDecoder(FunnelEncoder):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionUpsampleStructure(config) if config.use_attention_upsample else FunnelAttentionStructure(config)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList([FunnelLayer(config, block_index) for _ in range(block_size)])
                for block_index, block_size in enumerate(config.block_sizes)
            ]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        skip_connection_wieghts=None
    ):
        if skip_connection_wieghts is None:
            skip_connection_wieghts = torch.zeros(len(self.blocks))

        # The upsampling is not implemented on long tensors, so we convert this mask.
        attention_mask = attention_mask.type_as(hidden_states)
        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden_states,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = hidden_states

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        upsampled_hidden = None

        for block_index, block in enumerate(self.blocks):
            upsampling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            upsampling_flag = upsampling_flag and block_index > 0
            if upsampling_flag:
                upsampled_hidden, attention_inputs = self.attention_structure.pre_attention_upsampling(
                    hidden, attention_inputs
                )
            for (layer_index, layer) in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_upsampling = (repeat_index == 0) and (layer_index == 0) and upsampling_flag
                    if do_upsampling:
                        query = upsampled_hidden
                        key = value = hidden if self.config.upsample_q_only else upsampled_hidden
                    else:
                        query = key = value = hidden
                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions)
                    hidden = layer_output[0]
                    if do_upsampling:
                        attention_inputs = self.attention_structure.post_attention_upsampling(attention_inputs)

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


class FunnelAePreTrainedModel(FunnelPreTrainedModel):

    def _init_weights(self, module):
        classname = module.__class__.__name__
        if classname.find("Linear") != -1:
            if getattr(module, "weight", None) is not None:
                if self.config.initializer_std is None:
                    fan_out, fan_in = module.weight.shape
                    std = np.sqrt(1.0 / float(fan_in + fan_out))
                else:
                    std = self.config.initializer_std
                nn.init.normal_(module.weight, std=std)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0.0)
        elif classname == "FunnelRelMultiheadAttention":
            nn.init.uniform_(module.r_w_bias, b=self.config.initializer_range)
            nn.init.uniform_(module.r_r_bias, b=self.config.initializer_range)
            nn.init.uniform_(module.r_kernel, b=self.config.initializer_range)
            nn.init.uniform_(module.r_s_bias, b=self.config.initializer_range)
            nn.init.uniform_(module.seg_embed, b=self.config.initializer_range)
        elif classname == "FunnelEmbeddings":
            std = 1.0 if self.config.initializer_std is None else self.config.initializer_std
            nn.init.normal_(module.word_embeddings.weight, std=std)
            if module.word_embeddings.padding_idx is not None:
                module.word_embeddings.weight.data[module.padding_idx].zero_()


class FunnelAeBaseModel(FunnelAePreTrainedModel):
    config_class = FunnelAeConfig
    base_model_prefix = "ae"

    def __init__(self, config: Optional[PretrainedConfig]):
        assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        super().__init__(config)

        self.encoder = FunnelEncoder(config)
        self.decoder = FunnelAeDecoder(config)

        self.skip_connection_wieghts = torch.zeros(self.config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the TransformerVae directly is not supported."
            "Please use the respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # TODO: deal with head_mask
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        encoder_hidden_states = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        decoder_outputs = self.decoder(
            encoder_hidden_states,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            skip_connection_wieghts=self.skip_connection_wieghts
        )

        if not return_dict:
            idx = 0
            outputs = (decoder_outputs[0],)
            if output_hidden_states:
                idx += 1
                outputs = outputs + (encoder_hidden_states[1] + decoder_outputs[idx],)
            if output_attentions:
                idx += 1
                outputs = outputs + (encoder_hidden_states[2] + decoder_outputs[idx],)
            return outputs

        return BaseModelOutput(
            last_hidden_state=decoder_outputs[0],
            hidden_states=(encoder_hidden_states.hidden_states + decoder_outputs.hidden_states)
            if output_hidden_states
            else None,
            attentions=(encoder_hidden_states.attentions + decoder_outputs.attentions) if output_attentions else None,
        )


class FunnelAeForAutoencoding(FunnelPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.funnel = FunnelAeBaseModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]
        prediction_logits = self.lm_head(last_hidden_state)

        auto_enc_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            auto_enc_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_logits,) + outputs[1:]
            return ((auto_enc_loss,) + output) if auto_enc_loss is not None else output

        return AutoEncOutput(
            loss=auto_enc_loss,
            logits=prediction_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
