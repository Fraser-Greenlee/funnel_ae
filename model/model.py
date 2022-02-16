import numpy as np
import torch
from typing import Dict, Optional
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

from transformers.utils import logging
from transformers.models.funnel.modeling_funnel import (
    upsample, FunnelAttentionStructure, FunnelLayer, FunnelEmbeddings
)
from transformers.modeling_outputs import BaseModelOutput
from transformers import (
    FunnelPreTrainedModel,
    PretrainedConfig,
)

from model.config import FunnelAeConfig
from model.outputs import AutoEncOutput, TrackAttentionInputsOutput


logger = logging.get_logger(__name__)

'''
class FunnelAttentionUpsampleStructure(nn.Module):
    """
    Contains helpers for `FunnelRelMultiheadAttention`.
    """

    cls_token_type_id: int = 2

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Track where we are at in terms of upsampling from the original input, e.g., by how much the sequence length was
        # multiplied.
        self.upsampling_mult = None

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

        breakpoint()

        return tensor
'''

class FunnelEncoderUseSavedAttnInputs(nn.Module):
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
        inputs_embeds,
        all_attention_inputs: Dict[tuple, Tensor],
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        hidden = inputs_embeds

        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block_index, block in enumerate(self.blocks):
            pooling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                pooled_hidden = self.attention_structure.pool_tensor(hidden, mode=self.config.pooling_type)
            for (layer_index, layer) in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden if self.config.pool_q_only else pooled_hidden
                    else:
                        query = key = value = hidden

                    layer_output = layer(
                        query, key, value, all_attention_inputs[(block_index, layer_index)], output_attentions=output_attentions
                    )
                    hidden = layer_output[0]

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


class FunnelAeDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList([FunnelLayer(config, block_index) for _ in range(block_size)])
                for block_index, block_size in enumerate(config.block_sizes)
            ]
        )

    def upsample_hidden(self, hidden):
        '''
            Upsample hidden by repeating tokens to get a target length.
            Then average the repeated tokens with their neighbours.
        '''
        hidden = upsample(hidden, stride=2, target_len=hidden.size(1)*2, separate_cls=self.config.separate_cls)
        odd = hidden[:,1::2]
        even = hidden[:,::2] if hidden.size(1) % 2 == 0 else hidden[:,:-1:2]
        new_odd = (even + odd)/2
        hidden[:,1::2] = new_odd
        return hidden

    def forward(
        self,
        hidden_states,
        all_attention_inputs: Dict[tuple, Tensor],
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        skip_connection_wieghts=None
    ):
        if skip_connection_wieghts is None:
            skip_connection_wieghts = torch.zeros(len(self.blocks))

        hidden = hidden_states[-1]

        all_hidden_states = hidden_states if output_hidden_states else None
        all_attentions = () if output_attentions else None
        upsampled_hidden = None

        for block_index, block in list(enumerate(self.blocks))[::-1]:
            upsampling_flag = hidden.size(1) < all_hidden_states[0].size(1)
            upsampling_flag = upsampling_flag and block_index < len(self.blocks)-1
            if upsampling_flag:
                upsampled_hidden = self.upsample_hidden(hidden)
            for (layer_index, layer) in list(enumerate(block))[::-1]:
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_upsampling = (repeat_index == 0) and (layer_index == 0) and upsampling_flag
                    if do_upsampling:
                        query = upsampled_hidden
                        key = value = hidden if self.config.upsample_q_only else upsampled_hidden
                    else:
                        query = key = value = hidden

                    layer_output = layer(query, key, value, all_attention_inputs[(block_index, layer_index)], output_attentions=output_attentions)
                    hidden = layer_output[0]

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


class FunnelAutoEncAttentionStructure(FunnelAttentionStructure):
    def get_all_attention_inputs(self, inputs_embeds, attention_mask=None, token_type_ids=None):
        attention_inputs = self.init_attention_inputs(
            inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        all_attention_inputs = {}

        hidden = inputs_embeds
        for block_index, block in enumerate(self.blocks):
            pooling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )
            for (layer_index, _layer) in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    if do_pooling:
                        hidden = pooled_hidden

                    all_attention_inputs[(block_index, layer_index)] = attention_inputs

                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)

        return all_attention_inputs


class FunnelAeBaseModel(FunnelAePreTrainedModel):
    config_class = FunnelAeConfig
    base_model_prefix = "ae"

    def __init__(self, config: Optional[PretrainedConfig]):
        assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        super().__init__(config)
        self.config = config
        self.embeddings = FunnelEmbeddings(config)
        self.encoder = FunnelEncoderUseSavedAttnInputs(config)
        self.decoder = FunnelAeDecoder(config)
        self.attention_structure = FunnelAutoEncAttentionStructure(config)

        self.skip_connection_wieghts = torch.zeros(len(self.config.block_sizes))

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

    def get_output_embeddings(self):
        return self.get_input_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.set_input_embeddings(new_embeddings)

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
        # TODO allow recieving a latent code
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

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # TODO hanlde all attention inputs separately
        seq_len = inputs_embeds.size(1)

        all_attention_inputs = self.attention_structure.get_all_attention_inputs(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        encoder_outputs = self.encoder(
            inputs_embeds, all_attention_inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        encoder_hidden_states, all_attention_inputs = encoder_outputs[0], encoder_outputs[-1]

        breakpoint()

        decoder_outputs = self.decoder(
            encoder_hidden_states,
            (position_embeds, ),
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

        self.funnel_ae = FunnelAeBaseModel(config)
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

        outputs = self.funnel_ae(
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
