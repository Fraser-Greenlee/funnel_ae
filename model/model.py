from dataclasses import dataclass
from typing import List, Tuple
from torch import nn
import torch
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import BaseModelOutput
from transformers.models.funnel.modeling_funnel import (
    FunnelEmbeddings, FunnelEncoder, FunnelLayer, FunnelAttentionStructure, FunnelRelMultiheadAttention
)
from transformers import (
    FunnelPreTrainedModel, FunnelBaseModel
)

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

    def upsample_tensor(self, tensor, mode="mean", stride=2):
        """Apply 1D upsamping to a tensor of size [B x T (x H)]."""
        # make copies of tokens with mean interpolation
        z = 1
        pass


@dataclass
class AttentionInputsOutput(BaseModelOutput):
    all_attention_inputs: List[Tuple[torch.tensor]] = None


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
        all_attention_inputs = [attention_inputs]
        hidden = inputs_embeds

        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block_index, block in enumerate(self.blocks):
            pooling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )
                all_attention_inputs.append(attention_inputs)
            for (layer_index, layer) in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden if self.config.pool_q_only else pooled_hidden
                    else:
                        query = key = value = hidden
                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions)
                    hidden = layer_output[0]
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)
                        all_attention_inputs.append(attention_inputs)

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_attention_inputs, all_hidden_states, all_attentions] if v is not None)
        return AttentionInputsOutput(last_hidden_state=hidden, attention_inputs=all_attention_inputs, hidden_states=all_hidden_states, attentions=all_attentions)



class FunnelAeDecoder(FunnelEncoder):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.attention_structure = FunnelUpsamplingAttentionStructure(config)

    def forward(
        self,
        last_hidden_state,
        encoder_hidden_states=None,
        all_attention_inputs=None,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # The pooling is not implemented on long tensors, so we convert this mask.
        attention_mask = attention_mask.type_as(last_hidden_state)
        if all_attention_inputs is None:
            all_attention_inputs = self.attention_structure.get_all_attention_inputs(
                last_hidden_state,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        attention_inputs = all_attention_inputs.pop()
        hidden = last_hidden_state

        all_hidden_states = (last_hidden_state,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # TODO add residual with `encoder_hidden_states`

        for block_index, block in enumerate(self.blocks):
            upsampling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            upsampling_flag = upsampling_flag and block_index > 0
            if upsampling_flag:
                upsampled_hidden = self.attention_structure.upsample_tensor(hidden, mode=self.config.upsampling_type)
                attention_inputs = all_attention_inputs.pop()
            for (layer_index, layer) in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_upsampling = (repeat_index == 0) and (layer_index == 0) and upsampling_flag
                    if do_upsampling:
                        query = upsampled_hidden
                        # TODO impliment upsampling attention
                        key = value = upsampled_hidden
                    else:
                        query = key = value = hidden
                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions)
                    hidden = layer_output[0]
                    if do_upsampling:
                        attention_inputs = all_attention_inputs.pop()

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


class FunnelAeBaseModel(FunnelBaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = FunnelEmbeddings(config)
        self.encoder = FunnelEncoder(config)
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
        last_hidden_state, attention_inputs, hidden_states = encoder_outputs[:3]

        decoder_outputs = self.decoder(
            last_hidden_state,
            encoder_hidden_states=hidden_states,
            all_attention_inputs=attention_inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return decoder_outputs + encoder_outputs


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
