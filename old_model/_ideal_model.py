from collections import OrderedDict
from dataclasses import dataclass
from turtle import forward
from typing import List, Tuple
import numpy as np
from torch import nn
import torch
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.models.funnel.modeling_funnel import (
    FunnelEmbeddings, FunnelLayer, FunnelAttentionStructure, FunnelRelMultiheadAttention
)
from transformers import (
    FunnelPreTrainedModel, FunnelBaseModel
)
from model import outputs
from model.config import FunnelAeConfig
from model.outputs import AutoEncOutput

@dataclass
class FunnelModelOutput(ModelOutput):
    last_hidden_state:      List[Tuple[torch.tensor]] = None
    decoder_hidden_states:  List[Tuple[torch.tensor]] = None
    decoder_attentions:     List[Tuple[torch.tensor]] = None
    encoder_hidden_states: List[Tuple[torch.tensor]] = None
    encoder_attentions:     List[Tuple[torch.tensor]] = None

class FunnelAePreTrainedModel(FunnelPreTrainedModel):

    config_class = FunnelAeConfig
    base_model_prefix = "funnel_ae"

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
                module.word_embeddings.weight.data[module.word_embeddings.padding_idx].zero_()



@dataclass
class ModelInputs(OrderedDict):
    input_ids: torch.tensor = None
    inputs_embeds: torch.tensor = None
    position_ids: torch.tensor = None
    token_type_ids: torch.tensor = None
    attention_mask: torch.tensor = None
    head_mask: torch.tensor = None

    def __post_init__(self, model):
        if self.input_ids is not None and self.inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif self.input_ids is not None:
            input_shape = self.input_ids.size()
        elif self.inputs_embeds is not None:
            input_shape = self.inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = self.input_ids.device if self.input_ids is not None else self.inputs_embeds.device

        if self.attention_mask is None:
            self.attention_mask = torch.ones(input_shape, device=device)
        if self.token_type_ids is None:
            self.token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if self.inputs_embeds is None:
            self.inputs_embeds = self.embeddings(self.input_ids)



@dataclass
class ModelOutputArgs(OrderedDict):
    output_attentions: bool = None
    output_hidden_states: bool = None
    return_dict: bool = None

    def __post_init__(self, model):
        self.output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
        self.output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
        )
        self.return_dict = return_dict if return_dict is not None else model.config.use_return_dict



@dataclass
class FunnelModelInputs(ModelInputs):
    skip_w: List[float] = None

    def _pad_inputs(self):
        input_ids = self.input_ids
        if input_ids is not None and input_ids.size(1) % 2 == 1:
            pad = torch.tensor([self.config.pad_token_id])
            self.input_ids = torch.cat((input_ids, pad.expand(input_ids.size(0), 1)), axis=1)

        attention_mask = self.attention_mask
        if attention_mask is not None and attention_mask.size(1) % 2 == 1:
            pad = torch.tensor([0])
            self.attention_mask = torch.cat((attention_mask, pad.expand(attention_mask.size(0), 1)), axis=1)

        inputs_embeds = self.inputs_embeds
        if inputs_embeds is not None and inputs_embeds.size(1) % 2 == 1:
            pad = torch.tensor([self.config.pad_token_id])
            pad_embed = self.embeddings(pad)
            self.inputs_embeds = torch.cat((inputs_embeds, pad_embed.expand(inputs_embeds.size(0), 1, inputs_embeds.size(2))), axis=1)

        position_ids = self.position_ids
        if position_ids is not None and position_ids.size(1) % 2 == 1:
            breakpoint()

        token_type_ids = self.token_type_ids
        if token_type_ids is not None and token_type_ids.size(1) % 2 == 1:
            pad = torch.tensor([0])
            # TODO is this the way to pad token_type_ids?
            self.token_type_ids = torch.cat((token_type_ids, pad.expand(token_type_ids.size(0), 1)), axis=1)

    def __post_init__(self, model):
        super().__post_init__(model)
        self._pad_inputs()


class TransformerBase:
    def __call__(self, *args, **kwargs):
        arg_dataclasses = argparse(self.forward)
        args = parse(self, arg_dataclasses, *args, **kwargs)
        return self.forward(args)


@dataclass
class AttentionInputs(OrderedDict):
    position_embeds: torch.tensor
    token_type_mat: torch.tensor
    attention_mask: torch.tensor
    cls_mask: torch.tensor = None

    def pre_pooling(self, output):
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

    def post_pooling(self, attention_inputs):
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


class FunnelBlock(nn.Module):
    def __init__(self, config, block_index):
        super().__init__()
        self.config = config
        self.block_index = block_index
        self.block_size = config.block_sizes[block_index]
        self.repeats = config.block_repeats[block_index]
        self.run_pooling = block_index > 0
        self.layers = nn.ModuleList([
                FunnelLayer(config, block_index) for _ in range(self.block_size)
            ]
        )

    def forward(self, hidden: torch.tensor, attention_inputs: AttentionInputs, output_args: ModelOutputArgs):
        skip_hidden_state = None
        if self.run_pooling:
            assert hidden.size(1) > (2 if self.config.separate_cls else 1)
            skip_hidden_state = hidden
            pooled_hidden, attention_inputs = attention_inputs.pre_pooling(hidden)

        all_attentions = () if output_args.output_attentions else None
        all_hidden_statess = () if output_args.output_hidden_states else None

        for (layer_index, layer) in enumerate(self.layers): # TODO: enumerate(self.layers) or enumerate(self) ?
            for repeat_index in range(self.repeats):
                do_pooling = (repeat_index == 0) and (layer_index == 0) and self.run_pooling
                if do_pooling:
                    query = pooled_hidden
                    key = value = hidden if self.config.pool_q_only else pooled_hidden
                else:
                    query = key = value = hidden
                layer_output = layer(query, key, value, attention_inputs, output_attentions=output_args.output_attentions)
                hidden = layer_output[0]
                if do_pooling and self.config.pool_q_only:
                    attention_inputs = attention_inputs.post_pooling()

                if output_args.output_attentions:
                    all_attentions = all_attentions + layer_output[1:]
                if output_args.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden,)

        return BlockOutput(
            hidden, attention_inputs, skip_hidden_state, all_attentions, all_hidden_statess
        )


class FunnelAeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionStructure(config)
        self.blocks = nn.ModuleList([FunnelBlock(block_index) for block_index, in range(len(config.block_sizes))]

    def forward(self, inputs: ModelInputs, output_args: ModelOutputArgs):
        # The pooling is not implemented on long tensors, so we convert this mask.
        attention_mask = attention_mask.type_as(inputs_embeds)
        attention_inputs = self.attention_structure.init_attention_inputs(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        all_attention_inputs = {}
        hidden = inputs_embeds

        all_skip_hidden_states = ()
        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block_index, block in enumerate(self.blocks):
            block(hidden, output_args)

        breakpoint()

        if not return_dict:
            return tuple(v for v in [hidden, all_attention_inputs, all_skip_hidden_states, all_hidden_states, all_attentions] if v is not None)
        return AttentionInputsOutput(last_hidden_state=hidden, all_skip_hidden_states=all_skip_hidden_states, all_attention_inputs=all_attention_inputs, hidden_states=all_hidden_states, attentions=all_attentions)


class FunnelAeBaseModel(FunnelBaseModel, FunnelAePreTrainedModel, TransformerBase):

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = FunnelEmbeddings(config)
        self.encoder = FunnelAeEncoder(config)
        self.decoder = FunnelAeDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, inputs: FunnelModelInputs, output_args: ModelOutputArgs):
        encoder_outputs = self.encoder(inputs, output_args)
        decoder_outputs = self.decoder(encoder_outputs, inputs, output_args)
        return FunnelModelOutput(
            encoder_outputs,
            decoder_outputs,
            output_args
        )


class FunnelAeForAutoencoding(FunnelAePreTrainedModel):

    config_class = FunnelAeConfig
    base_model_prefix = "funnel_ae"

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
        skip_w=None,
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
            skip_w=skip_w,
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
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,

        )