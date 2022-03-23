import copy
from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput, ModelOutput, MaskedLMOutput
from transformers.models.funnel.modeling_funnel import (
    FunnelLayer, FunnelAttentionStructure, FunnelPositionwiseFFN, ACT2FN
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
        config = copy.deepcopy(config)
        self.config = config
        config.pool_q_only = config.upsample_q_only
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
        if config.upsample_mode == "ff_seperator":
            self.seperators = nn.ModuleList([
                FunnelPositionwiseFFN(config) for _ in config.block_sizes
            ])
        elif config.upsample_mode == "lin_seperator":
            self.seperators = nn.ModuleList([
                nn.ModuleList([nn.Linear(config.d_model, config.d_model), nn.LayerNorm(config.d_model, config.layer_norm_eps)])
                for _ in config.block_sizes
            ])
        elif config.upsample_mode == "double_avg":
            pass
        elif config.upsample_mode:
            raise NotImplementedError(f'Not implimeneted `config.upsample_mode`={config.upsample_mode}')

    @staticmethod
    def avg_hidden(hidden1, hidden2):
        interpolated_hidden = torch.mean(torch.stack([hidden1, hidden2], axis=1), axis=1)
        return interpolated_hidden

    def _upsample_hidden(self, hidden, block_index):
        seq_len = hidden.size(1)

        if self.config.upsample_mode:
            seperators = self.seperators[block_index](hidden)
            cat_hidden = torch.cat((hidden - seperators, hidden + seperators), axis=1)
        else:
            # make new tokens by averaging their adjacents
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

    def upsample_hidden(self, hidden, target_len, block_index):

        if self.config.separate_cls:
            cls_token = hidden[:,:1,:]
            doubled_hidden = self._upsample_hidden(hidden[:,1:,:], block_index)

            end_token = []
            if doubled_hidden.size(1) + 1 == target_len - 1:
                cls_w_last = self.avg_hidden(cls_token, doubled_hidden[:,-1:,:])
                end_token = [cls_w_last]

            doubled_hidden = torch.cat([cls_token, doubled_hidden] + end_token, axis=1)

        else:
            doubled_hidden = self._upsample_hidden(hidden, block_index)

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
        hidden = block_hiddens[0]
        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # run the same encoder steps to get attention inputs
        all_attention_inputs = {}
        pooled_hidden = None
        for block_index, block_size in enumerate(self.config.block_sizes):
            pooling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )
            for layer_index in range(block_size):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    # run layer
                    if pooled_hidden is not None:
                        hidden = pooled_hidden
                    all_attention_inputs[block_index, layer_index, repeat_index] = attention_inputs
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)

        hidden = final_hidden
        all_hidden_states = (hidden,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # run the decoder
        for block_index, block in list(enumerate(self.blocks))[::-1]:
            # TODO allow skip connections from sampled latent tokens
            # if latent block index:
            #     hidden += latent_hidden[block_index] 
            upsampling_flag = hidden.size(1) < block_hiddens[0].size(1)
            upsampling_flag = upsampling_flag and block_index > 0
            for (layer_index, layer) in list(enumerate(block))[::-1]:
                for repeat_index in range(self.config.block_repeats[block_index])[::-1]:
                    do_upsampling = (repeat_index == 0) and (layer_index == 0) and upsampling_flag
                    query = key = value = hidden

                    if do_upsampling and self.config.upsample_q_only:
                        # TODO I don't think the attention mask will be the right size here, needs to be the last one used (or just pool it)
                        query = self.upsample_hidden(query, target_size, block_index)
                        position_embeds, token_type_mat, attention_mask, cls_mask = all_attention_inputs[block_index, layer_index, repeat_index]
                        cls_mask = None if cls_mask is None else cls_mask.T
                        token_type_mat = token_type_mat.permute(0,2,1)
                        attention_inputs = position_embeds, token_type_mat, attention_mask, cls_mask
                    else:
                        attention_inputs = all_attention_inputs[block_index, layer_index, repeat_index]

                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions)
                    hidden = layer_output[0]

                    if do_upsampling and not self.config.upsample_q_only:
                        target_size = block_hiddens[block_index-1].size(1)
                        hidden = self.upsample_hidden(hidden, target_size, block_index)

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


class PositionwiseLatentFFN(nn.Module):
    def __init__(self, config, is_encoder=True):
        super().__init__()
        self.d_in = config.d_model if is_encoder else config.d_latent
        self.d_out = config.d_latent if is_encoder else config.d_model

        self.linear_1 = nn.Linear(self.d_in, config.d_inner)
        self.activation_function = ACT2FN[config.hidden_act]
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        self.linear_2 = nn.Linear(config.d_inner, self.d_out)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(self.d_out, config.layer_norm_eps)

    def forward(self, hidden):
        h = self.linear_1(hidden)
        h = self.activation_function(h)
        # TODO remove dropout?
        h = self.activation_dropout(h)
        h = self.linear_2(h)
        h = self.dropout(h)
        return self.layer_norm(h)


class MMD_VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # TODO try tanh
        self.enc = PositionwiseLatentFFN(config)
        self.dec = PositionwiseLatentFFN(config, is_encoder=False)

    @staticmethod
    def _compute_kernel(x, y):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]

        tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
        tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

        return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim * 1.0)

    def _compute_mmd(self, x, y):
        x_kernel = self._compute_kernel(x, x)
        y_kernel = self._compute_kernel(y, y)
        xy_kernel = self._compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def _reg_loss(self, latent):
        true_samples = torch.randn(latent.size(), device=latent.device)
        return self._compute_mmd(true_samples, latent)

    def reg_loss(self, latent):
        batch_size, n_latents_per_batch, latent_code_dim = latent.size()
        reg_loss = self._reg_loss(latent.reshape(-1, latent_code_dim)) / batch_size * n_latents_per_batch
        return reg_loss

    def forward(self, hidden):
        latent = self.enc(hidden)
        recon = self.dec(latent)
        reg_loss = self.reg_loss(latent)
        return recon, latent, reg_loss


class AE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # TODO try tanh
        self.enc = PositionwiseLatentFFN(config)
        self.dec = PositionwiseLatentFFN(config, is_encoder=False)

    def forward(self, hidden):
        latent = self.enc(hidden)
        recon = self.dec(latent)
        return recon, latent, None

class KL_VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mu = PositionwiseLatentFFN(config)
        self.var = PositionwiseLatentFFN(config)
        self.dec = PositionwiseLatentFFN(config, is_encoder=False)

    def reg_loss(self, mu, logvar):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return self.config.beta * kld_loss

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, hidden):
        mu, logvar = self.mu(hidden), self.var(hidden)
        latent = self.reparameterize(mu, logvar)
        recon = self.dec(latent)
        reg_loss = self.reg_loss(mu, logvar).mean()
        return recon, latent, reg_loss


VAEs = {'MMD': MMD_VAE, 'KL': KL_VAE, 'AE': AE, '': lambda *args: None}


class BaseVAEModelOutput(ModelOutput):
    reg_loss: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class FunnelAeModel(FunnelAePreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.base_funnel = FunnelBaseModel(config)
        self.embeddings = self.base_funnel.embeddings
        self.decoder = FunnelAeDecoder(config, encoder_blocks=self.base_funnel.encoder.blocks if config.share_encoder_blocks else None)
        self.vae = VAEs[self.config.vae](config)
        self.encoder_held_out_blocks = []
        self.decoder_held_out_blocks = []

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _cut_blocks(self, n, model_blocks, held_out_blocks):
        blocks = held_out_blocks + [block for block in model_blocks]
        return nn.ModuleList(blocks[n:]), blocks[:n]

    def cut_to_n_blocks(self, n):
        if n == 0:
            n = len(self.decoder.blocks)
        self.base_funnel.encoder.blocks, self.encoder_held_out_blocks = self._cut_blocks(
            n, self.base_funnel.encoder.blocks, self.encoder_held_out_blocks
        )
        self.decoder.blocks, self.decoder_held_out_blocks = self._cut_blocks(
            n, self.decoder.blocks, self.decoder_held_out_blocks
        )

    def n_blocks(self):
        return len(self.base_funnel.encoder.blocks)

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

        assert max(self.config.block_repeats) == 1 # breaks `block_hiddens`

        recon_hidden, latent, reg_loss = self.vae(encoder_outputs[0]) if self.vae is not None else (encoder_outputs[0], None, None)

        decoder_outputs = self.decoder(
            final_hidden=recon_hidden if not self.config._randn_enc else torch.randn_like(recon_hidden),
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

        return BaseVAEModelOutput(
            reg_loss=reg_loss,
            last_hidden_state=decoder_outputs[0],
            hidden_states=(encoder_outputs.hidden_states + ([latent, recon_hidden] if latent else []) + decoder_outputs.hidden_states)
            if output_hidden_states
            else None,
            attentions=(encoder_outputs.attentions + decoder_outputs.attentions) if output_attentions else None,
        )


class FunnelAeForMaskedLM(FunnelForMaskedLM):
    def __init__(self, config):
        super(FunnelForMaskedLM, self).__init__(config)

        self.funnel = FunnelAeModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.tr_masked_lm_loss = 0.0
        self.tr_reg_loss = 0.0

        # Initialize weights and apply final processing
        self.post_init()

    def get_masked_lm_loss(self):
        result = self.tr_masked_lm_loss
        self.tr_masked_lm_loss = 0.0
        return result

    def get_reg_loss(self):
        result = self.tr_reg_loss
        self.tr_reg_loss = 0.0
        return result

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

        last_hidden_state = outputs[1]
        prediction_logits = self.lm_head(last_hidden_state)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))

        reg_loss = outputs[0]

        if reg_loss is not None and masked_lm_loss is not None and reg_loss is not None:
            if self.training:
                self.tr_masked_lm_loss += masked_lm_loss
                self.tr_reg_loss += reg_loss
            masked_lm_loss += reg_loss

        if not return_dict:
            output = (prediction_logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
