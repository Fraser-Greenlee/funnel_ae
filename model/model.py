from dataclasses import dataclass
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
from model.config import FunnelAeConfig

from model.outputs import AutoEncOutput


class FunnelRelMultiheadAttentionUpsampling(FunnelRelMultiheadAttention):
    # TODO impliment upsampling attention
    pass


class FunnelUpsamplingAttentionStructure(FunnelAttentionStructure):
    '''
        Upsamples tokens by interpolating between adjacent tokens to create new ones.
    '''
    def get_full_seq_len(self, cmp_size):
        cmp_size -= 1 if self.config.separate_cls else 0
        if cmp_size <= 0:
            raise Exception('Not enough latent tokens.')
        n_upsamples = len(self.config.block_sizes) - 1
        decomp_size = cmp_size * 2 ** n_upsamples
        decomp_size += 1 if self.config.separate_cls else 0
        return decomp_size

    def pool_seq_len(self, seq_len):
        seq_len -= 1 if self.config.separate_cls else 0
        seq_len //= 2
        seq_len += 1 if self.config.separate_cls else 0
        return seq_len

    def pre_attention_pooling(self, attention_inputs):
        # modified to not use `output`
        """Pool the proper parts of `attention_inputs` before the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config.pool_q_only:
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds[:2], 0) + position_embeds[2:]
            token_type_mat = self.stride_pool(token_type_mat, 1)
            cls_mask = self.stride_pool(cls_mask, 0)
        else:
            self.pooling_mult *= 2
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds, 0)
            token_type_mat = self.stride_pool(token_type_mat, [1, 2])
            cls_mask = self.stride_pool(cls_mask, [1, 2])
            attention_mask = self.pool_tensor(attention_mask, mode="min")
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return attention_inputs

    def get_all_attention_inputs(self, cmp_embeds, attention_mask=None, token_type_ids=None):
        # NOT USED!
        """Returns the attention inputs associated to the inputs of the model."""
        # cmp_embeds has shape batch_size x compressed(seq_len) x d_model
        # attention_mask and token_type_ids have shape batch_size x seq_len
        all_attention_inputs = []
        self.seq_len = seq_len = self.get_full_seq_len(cmp_embeds.size(1))
        full_size_embeds = cmp_embeds.new_zeros((cmp_embeds.size(0), seq_len, cmp_embeds.size(2)))
    
        attention_inputs = self.init_attention_inputs(full_size_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)

        all_attention_inputs.append(attention_inputs)
        for _ in range(len(self.config.block_sizes))[:1]:
            attention_inputs = self.pre_attention_pooling(attention_inputs)
            all_attention_inputs.append(attention_inputs)
            attention_inputs = self.post_attention_pooling(attention_inputs)
            all_attention_inputs.append(attention_inputs)

        return all_attention_inputs

    def upsample_hidden(self, hidden):
        # make new tokens by averaging adjacent
        seq_len = hidden.size(1)
        token_ids = list(range(seq_len))
        shifted_token_ids = [token_ids[-1]] + token_ids[:-1]
        shifted_hidden = hidden[:,shifted_token_ids,:]
        # average these to get new tokens
        new_hidden = torch.mean(torch.stack([hidden, shifted_hidden], axis=1), axis=1)
        cat_hidden = torch.cat((hidden, new_hidden), axis=1)
        # shuffle so new hidden tokens are avg of adjacent
        mixed_ids = []
        for i in range(seq_len):
            mixed_ids += [i, i + seq_len]
        final_hidden = cat_hidden[:,mixed_ids,:]
        return final_hidden


@dataclass
class AttentionInputsOutput(BaseModelOutput):
    all_attention_inputs: List[Tuple[torch.tensor]] = None
    upsample_blocks: List[int] = None


class FunnelAeEncoder(nn.Module):
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
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # The pooling is not implemented on long tensors, so we convert this mask.
        attention_mask = attention_mask.type_as(inputs_embeds)
        attention_inputs = self.attention_structure.init_attention_inputs(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        all_attention_inputs = {}
        upsample_blocks = []
        hidden = inputs_embeds

        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block_index, block in enumerate(self.blocks):
            pooling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                upsample_blocks.append(block_index)
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
                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions)
                    all_attention_inputs[(block_index, layer_index)] = attention_inputs
                    hidden = layer_output[0]
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_attention_inputs, upsample_blocks, all_hidden_states, all_attentions] if v is not None)
        return AttentionInputsOutput(last_hidden_state=hidden, all_attention_inputs=all_attention_inputs, upsample_blocks=upsample_blocks, hidden_states=all_hidden_states, attentions=all_attentions)


class FunnelAeDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.attention_structure = FunnelUpsamplingAttentionStructure(config)
        # TODO wrong `block_index` causes `relative_positional_attention` to fail
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList([FunnelLayer(config, len(config.block_sizes) - block_index - 1) for _ in range(block_size)])
                for block_index, block_size in enumerate(config.block_sizes[::-1])
            ]
        )


    def forward(
        self,
        last_hidden_state,
        encoder_hidden_states=None,
        all_attention_inputs=None,
        upsample_blocks=None,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # The pooling is not implemented on long tensors, so we convert this mask.
        attention_mask = attention_mask.type_as(last_hidden_state)
        hidden = last_hidden_state

        all_hidden_states = (last_hidden_state,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # TODO add residual with `encoder_hidden_states`

        for block_index, block in enumerate(self.blocks):
            upsampling_flag = block_index in upsample_blocks
            if upsampling_flag:
                upsampled_hidden = self.attention_structure.upsample_hidden(
                    hidden[:,1:] if self.config.separate_cls else hidden
                )
                if self.config.separate_cls:
                    upsampled_hidden = torch.cat((hidden[:,:1], upsampled_hidden), axis=1)
            for (layer_index, layer) in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_upsampling = (repeat_index == 0) and (layer_index == 0) and upsampling_flag
                    if do_upsampling:
                        query = upsampled_hidden
                        # TODO impliment upsampling attention
                        key = value = upsampled_hidden
                    else:
                        query = key = value = hidden
                    try:
                        layer_output = layer(query, key, value, all_attention_inputs[(len(self.blocks) - block_index - 1, layer_index)], output_attentions=output_attentions)
                    except Exception:
                        breakpoint()
                    hidden = layer_output[0]

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


@dataclass
class FunnelModelOutput(ModelOutput):
    last_hidden_state:      List[Tuple[torch.tensor]] = None
    decoder_hidden_states:  List[Tuple[torch.tensor]] = None
    decoder_attentions:     List[Tuple[torch.tensor]] = None
    encoder_hidden_states: List[Tuple[torch.tensor]] = None
    encoder_attentions:     List[Tuple[torch.tensor]] = None


class FunnelAeBaseModel(FunnelBaseModel):

    config_class = FunnelAeConfig

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

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = FunnelEmbeddings(config)
        self.encoder = FunnelAeEncoder(config)
        self.decoder = FunnelAeDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def _base_forward(
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
    ):
        '''
            Just copies the start of FunnelBaseModel.forward()
        '''
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

        return (
            inputs_embeds,
            attention_mask,
            token_type_ids,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

    def forward(
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
    ):
        # TODO add padding support
        if input_ids is not None and input_ids.size(1) % 2 == 1:
            pad = torch.tensor([self.config.pad_token_id])
            input_ids = torch.cat((input_ids, pad.expand(input_ids.size(0), 1)), axis=1)

        if attention_mask is not None and attention_mask.size(1) % 2 == 1:
            pad = torch.tensor([0])
            attention_mask = torch.cat((attention_mask, pad.expand(attention_mask.size(0), 1)), axis=1)

        if inputs_embeds is not None and inputs_embeds.size(1) % 2 == 1:
            pad = torch.tensor([self.config.pad_token_id])
            pad_embed = self.embeddings(pad)
            inputs_embeds = torch.cat((inputs_embeds, pad_embed.expand(inputs_embeds.size(0), 1, inputs_embeds.size(2))), axis=1)

        if position_ids is not None and position_ids.size(1) % 2 == 1:
            breakpoint()

        if token_type_ids is not None and token_type_ids.size(1) % 2 == 1:
            pad = torch.tensor([0])
            # TODO is this the way to pad token_type_ids?
            token_type_ids = torch.cat((token_type_ids, pad.expand(token_type_ids.size(0), 1)), axis=1)

        (
            inputs_embeds,
            attention_mask,
            token_type_ids,
            output_attentions,
            output_hidden_states,
            return_dict
        ) = self._base_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state, hidden_states, attention_inputs, upsample_blocks = encoder_outputs[:4]

        decoder_outputs = self.decoder(
            last_hidden_state,
            encoder_hidden_states=hidden_states,
            all_attention_inputs=attention_inputs,
            upsample_blocks=upsample_blocks,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        kwargs = {}
        if output_attentions:
            kwargs['encoder_attentions'] = encoder_outputs[3]
            kwargs['decoder_attentions'] = decoder_outputs[2]

        return FunnelModelOutput(
            last_hidden_state=decoder_outputs[0],
            decoder_hidden_states=decoder_outputs[1],
            encoder_hidden_states=hidden_states,
            **kwargs
        )


class FunnelAeForAutoencoding(FunnelPreTrainedModel):

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
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,

        )
