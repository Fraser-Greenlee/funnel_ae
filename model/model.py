import numpy as np
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.funnel.modeling_funnel import (
    FunnelLayer, FunnelAttentionStructure, FunnelRelMultiheadAttention, _relative_shift_gather
)
from transformers import (
    FunnelBaseModel,
    FunnelPreTrainedModel,
    FunnelForMaskedLM,
)

INF = 1e6


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
            # Fix padding
            std = 1.0 if self.config.initializer_std is None else self.config.initializer_std
            nn.init.normal_(module.word_embeddings.weight, std=std)
            if module.word_embeddings.padding_idx is not None:
                module.word_embeddings.weight.data[module.word_embeddings.padding_idx].zero_()


class FunnelAeDecoder(nn.Module):
    def __init__(self, config, encoder_blocks=None):
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionStructure(config)
        if encoder_blocks is not None:
            self.blocks = encoder_blocks
        else:
            self.blocks = nn.ModuleList(
                [
                    nn.ModuleList([FunnelLayer(config, block_index) for _ in range(block_size)])
                    for block_index, block_size in enumerate(config.block_sizes)
                ]
            )
        self.skip_w = [0 for _ in config.block_sizes]

    @staticmethod
    def avg_hidden(hidden1, hidden2):
        interpolated_hidden = torch.mean(torch.stack([hidden1, hidden2], axis=1), axis=1)
        return interpolated_hidden

    def _upsample_hidden(self, hidden):
        # TODO allow upsample via FFN
        # make new tokens by averaging their adjacents
        seq_len = hidden.size(1)
        token_ids = list(range(seq_len))
        shifted_token_ids = [token_ids[-1]] + token_ids[:-1]
        shifted_hidden = hidden[:,shifted_token_ids,:]
        # average these to get new tokens
        interpolated_hidden = self.avg_hidden(hidden, shifted_hidden)
        cat_hidden = torch.cat((hidden, interpolated_hidden), axis=1)
        # shuffle so new hidden tokens are avg of adjacent
        mixed_ids = []
        for i in range(seq_len):
            mixed_ids += [i, i + seq_len]
        final_hidden = cat_hidden[:,mixed_ids,:]
        return final_hidden

    def upsample_hidden(self, hidden, target_len):

        if self.config.separate_cls:
            cls_token = hidden[:,:1,:]
            doubled_hidden = self._upsample_hidden(hidden[:,1:,:])

            end_token = []
            if doubled_hidden.size(1) + 1 == target_len - 1:
                cls_w_last = self.avg_hidden(cls_token, doubled_hidden[:,-1:,:])
                end_token = [cls_w_last]

            doubled_hidden = torch.cat([cls_token, doubled_hidden] + end_token, axis=1)

        else:
            doubled_hidden = self._upsample_hidden(hidden)

        # for odd number of target tokens, prune the interpolated token at the end
        if doubled_hidden.size(1) > target_len:
            doubled_hidden = doubled_hidden[:,:-1,:]

        return doubled_hidden

    def forward(
        self,
        final_hidden,
        block_hiddens,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # The pooling is not implemented on long tensors, so we convert this mask.
        attention_mask = attention_mask.type_as(block_hiddens[0])
        attention_inputs = self.attention_structure.init_attention_inputs(
            block_hiddens[0],
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # run the same encoder steps to get attention inputs
        all_attention_inputs = {}
        hidden = block_hiddens[0]
        for block_index, block_size in enumerate(self.config.block_sizes):
            pooling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                _pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )
            for layer_index in range(block_size):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    # run layer
                    all_attention_inputs[block_index, layer_index, repeat_index] = attention_inputs
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)

        hidden = final_hidden
        all_hidden_states = (hidden,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        last_attention_mask = None

        # run the decoder
        for block_index, block in list(enumerate(self.blocks))[::-1]:
            hidden = hidden * (1 - self.skip_w[block_index]) + self.skip_w[block_index] * block_hiddens[block_index]
            upsampling_flag = hidden.size(1) < block_hiddens[0].size(1)
            upsampling_flag = upsampling_flag and block_index > 0
            if upsampling_flag:
                target_size = block_hiddens[block_index-1].size(1)
                upsampled_hidden = self.upsample_hidden(hidden, target_size)
            for (layer_index, layer) in list(enumerate(block))[::-1]:
                for repeat_index in range(self.config.block_repeats[block_index])[::-1]:
                    do_upsampling = (repeat_index == 0) and (layer_index == 0) and upsampling_flag
                    if do_upsampling:
                        query = upsampled_hidden
                        key = value = hidden if self.config.pool_q_only else upsampled_hidden
                    else:
                        query = key = value = hidden

                    position_embeds, token_type_mat, attention_mask, cls_mask = all_attention_inputs[block_index, layer_index, repeat_index]

                    cls_mask = None if cls_mask is None else cls_mask.T
                    token_type_mat = token_type_mat.permute(0,2,1)

                    if query.shape[1] > key.shape[1]:
                        # if doing query only upsampling, use the previous 1/2 len attention mask
                        hold = attention_mask
                        attention_mask = last_attention_mask
                        last_attention_mask = hold
                    else:
                        last_attention_mask = attention_mask

                    attention_inputs = position_embeds, token_type_mat, attention_mask, cls_mask

                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions)
                    hidden = layer_output[0]

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


class FunnelAeModel(FunnelAePreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.base_funnel = FunnelBaseModel(config)
        self.embeddings = self.base_funnel.embeddings
        self.decoder = FunnelAeDecoder(config, encoder_blocks=self.base_funnel.encoder.blocks if config.share_encoder_blocks else None)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

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
        encoder_outputs = self.base_funnel.forward(
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
    
        # Copy Input handling from `FunnelBaseModel`
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

        decoder_outputs = self.decoder(
            final_hidden=encoder_outputs[0] if not self.config._randn_enc else torch.randn_like(encoder_outputs[0]),
            # TODO add support for `config.block_repeats`
            block_hiddens=[
                encoder_outputs[1][
                    sum(self.config.block_sizes[:i]) + block_size
                ]
                for i, block_size in enumerate(self.config.block_sizes)
            ],
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            idx = 0
            outputs = (decoder_outputs[0],)
            if output_hidden_states:
                idx += 1
                outputs = outputs + (encoder_outputs[1] + decoder_outputs[idx],)
            if output_attentions:
                idx += 1
                outputs = outputs + (encoder_outputs[2] + decoder_outputs[idx],)
            return outputs

        return BaseModelOutput(
            last_hidden_state=decoder_outputs[0],
            hidden_states=(encoder_outputs.hidden_states + decoder_outputs.hidden_states)
            if output_hidden_states
            else None,
            attentions=(encoder_outputs.attentions + decoder_outputs.attentions) if output_attentions else None,
        )


class FunnelAeForMaskedLM(FunnelForMaskedLM):
    def __init__(self, config):
        super(FunnelForMaskedLM, self).__init__(config)

        self.funnel = FunnelAeModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()
