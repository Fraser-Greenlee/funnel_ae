import torch
from typing import Optional
from torch import nn

from transformers.utils import logging
from transformers.models.funnel.modeling_funnel import (
    FunnelEncoder, FunnelAttentionStructure, FunnelLayer, upsample
)
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    BaseModelOutput
)

from model.config import FunnelAeConfig


logger = logging.get_logger(__name__)


class FunnelAttentionUpsampleStructure(FunnelAttentionStructure):
    def __init__(self, config):
        super().__init__(config)
        # Track where we are at in terms of pooling from the original input, e.g., by how much the sequence length was
        # divided.
        del self.pooling_mult
        self.upsampling_mult = None

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
            pooled_pos_id = pos_id[1:-1] if self.config.truncate_seq else pos_id[1:]
            return torch.cat([cls_pos, pooled_pos_id[::2]], 0)
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

        # Do the stride pool recursively if axis is a list or a tuple of ints.
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                tensor = self.stride_upsample(tensor, ax)
            return tensor

        # Do the stride pool recursively if tensor is a list or tuple of tensors.
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
        return tensor[enc_slice]

    def upsample_tensor(self, tensor, mode="mean", stride=2):
        """Apply 1D upsample to a tensor of size [B x T (x H)]."""
        # TODO
        if tensor is None:
            return None

        # Do the upsample recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.upsample_tensor(tensor, mode=mode, stride=stride) for x in tensor)

        if self.config.separate_cls:
            suffix = tensor[:, :-1] if self.config.truncate_seq else tensor
            tensor = torch.cat([tensor[:, :1], suffix], dim=1)

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

    def pre_attention_upsampling(self, output, attention_inputs):
        """Upsample `output` and the proper parts of `attention_inputs` before the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config.pool_q_only:
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_upsample(position_embeds[:2], 0) + position_embeds[2:]
            token_type_mat = self.stride_upsample(token_type_mat, 1)
            cls_mask = self.stride_upsample(cls_mask, 0)
            output = self.upsample_tensor(output, mode=self.config.pooling_type)
        else:
            self.upsampling_mult *= 2
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_upsample(position_embeds, 0)
            token_type_mat = self.stride_upsample(token_type_mat, [1, 2])
            cls_mask = self.stride_upsample(cls_mask, [1, 2])
            attention_mask = self.upsample_tensor(attention_mask, mode="min")
            output = self.upsample_tensor(output, mode=self.config.pooling_type)
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return output, attention_inputs

    def post_attention_upsampling(self, attention_inputs):
        """Upsample the proper parts of `attention_inputs` after the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config.pool_q_only:
            self.upsampling_mult *= 2
            if self.config.attention_type == "factorized":
                position_embeds = position_embeds[:2] + self.stride_upsample(position_embeds[2:], 0)
            token_type_mat = self.stride_upsample(token_type_mat, 2)
            cls_mask = self.stride_upsample(cls_mask, 1)
            attention_mask = self.upsample_tensor(attention_mask, mode="min")
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return attention_inputs


class FunnelAeDecoder(FunnelEncoder):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionStructure(config)
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

        # The pooling is not implemented on long tensors, so we convert this mask.
        attention_mask = attention_mask.type_as(hidden_states)
        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden_states,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = hidden_states

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block_index, block in enumerate(self.blocks):
            upsample_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            upsample_flag = upsample_flag and block_index > 0
            # TODO can I pre_attention_upsample?
            # TODO use skip_connection_wieghts[block_index]
            breakpoint()
            if upsample_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_upsample(
                    hidden, attention_inputs
                )
            for (layer_index, layer) in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_upsample = (repeat_index == 0) and (layer_index == 0) and upsample_flag
                    if do_upsample:
                        query = pooled_hidden
                        key = value = hidden if self.config.pool_q_only else pooled_hidden
                    else:
                        query = key = value = hidden
                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions)
                    hidden = layer_output[0]
                    if do_upsample:
                        attention_inputs = self.attention_structure.post_attention_upsample(attention_inputs)

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


class FunnelAe(PreTrainedModel):
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
