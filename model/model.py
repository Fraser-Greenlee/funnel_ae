import numpy as np
from torch import nn
from transformers import (
    FunnelBaseModel,
    FunnelPreTrainedModel,
    BaseModelOutput,
    FunnelForMaskedLM,
    FunnelAttentionStructure,
    FunnelLayer
)


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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionStructure(config)
        config.decoder_block_sizes = config.block_sizes[::-1]
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList([FunnelLayer(config, block_index) for _ in range(block_size)])
                for block_index, block_size in enumerate(config.decoder_block_sizes)
            ]
        )
        self.skip_w = [0 for _ in config.decoder_block_sizes]

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
        hidden = block_hiddens[block_index]
        for block_index, block in enumerate(self.blocks):
            pooling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                _pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )
            for (layer_index, layer) in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    # run layer
                    all_attention_inputs[block_index, layer_index, repeat_index] = attention_inputs
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)

        hidden = final_hidden
        all_hidden_states = (hidden,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # run the decoder
        for block_index, block in enumerate(self.blocks):
            hidden = hidden * (1 - self.skip_w[block_index]) + self.skip_w[block_index] * block_hiddens[block_index]

            inv_block_index = len(self.blocks) - block_index - 1
            upsampling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            upsampling_flag = upsampling_flag and block_index > 0
            if upsampling_flag:
                upsampled_hidden = self.attention_structure.upsample_tensor(hidden)
            for (layer_index, layer) in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_upsampling = (repeat_index == 0) and (layer_index == 0) and upsampling_flag
                    if do_upsampling:
                        query = upsampled_hidden
                        key = value = hidden if self.config.pool_q_only else upsampled_hidden
                    else:
                        query = key = value = hidden
                    layer_output = layer(query, key, value, all_attention_inputs[inv_block_index, layer_index, repeat_index], output_attentions=output_attentions)
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
        self.default_funnel = FunnelBaseModel(config)
        self.embeddings = self.default_funnel.embeddings
        self.decoder = FunnelAeDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

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
        encoder_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        decoder_outputs = self.decoder(
            final_hidden=encoder_outputs[0],
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
        super(FunnelForMaskedLM).__init__(config)

        self.funnel = FunnelAeModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()
