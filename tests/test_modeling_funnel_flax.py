import unittest

import jax
import jax.numpy as jnp

from transformers import FunnelConfig, FunnelTokenizer

from model.modeling_funnel_flax import (
    FunnelEmbeddings,
    FunnelAttentionStructure
)


tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/small")
input_ids = tokenizer("Hello there", return_tensors="np").input_ids
rng = jax.random.PRNGKey(0)


class TestEmbeddings(unittest.TestCase):
    def test_eval(self):
        config = FunnelConfig()
        emb = FunnelEmbeddings(config)
        variables = emb.init(rng, input_ids)
        embeddings = emb.apply(variables, input_ids)
        self.assertEqual(embeddings.shape, (1, input_ids.shape[-1], config.d_model))

    def test_train(self):
        config = FunnelConfig()
        emb = FunnelEmbeddings(config)
        variables = emb.init(rng, input_ids)
        embeddings = emb.apply(variables, input_ids, deterministic=False, rngs={'dropout': rng})
        self.assertEqual(embeddings.shape, (1, input_ids.shape[-1], config.d_model))


class TestFunnelAttentionStructure(unittest.TestCase):
    def test_init_attention_inputs(self):
        batch_size, seq_len = 2, 4
        config = FunnelConfig()
        inputs_embeds  = jax.random.normal(rng, (batch_size, seq_len, config.d_model))
        attention_mask = jnp.array([ [1,1,1,0], [1,1,1,1] ])
        token_type_ids = jnp.array([ [2,1,1,1], [2,2,1,1] ])
        attention_structure = FunnelAttentionStructure(config)
        variables = attention_structure.init(
            rng, inputs_embeds, attention_mask, token_type_ids, method=attention_structure.init_attention_inputs
        )
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_structure.apply(
            variables, inputs_embeds, attention_mask, token_type_ids, method=attention_structure.init_attention_inputs
        )
        self.assertEqual(
            [ll.shape if ll is not None else None for l in position_embeds for ll in l],
            [(8, 768), None, (4, 768), (8, 768), (2, 768), (4, 768)]
        )
        self.assertEqual(token_type_mat.size, token_type_mat.sum())
        self.assertEqual(attention_mask.tolist(), [[1, 1, 1, 0], [1, 1, 1, 1]])
        self.assertEqual(cls_mask.tolist(), [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0]])

    def test_pre_attention_pooling(self):
        batch_size, seq_len = 2, 4
        config = FunnelConfig()
        inputs_embeds  = jax.random.normal(rng, (batch_size, seq_len, config.d_model))
        attention_mask = jnp.array([ [1,1,1,0], [1,1,1,1] ])
        token_type_ids = jnp.array([ [2,1,1,1], [2,2,1,1] ])
        attention_structure = FunnelAttentionStructure(config)
        variables = attention_structure.init(
            rng, inputs_embeds, attention_mask, token_type_ids, method=attention_structure.init_attention_inputs
        )
        attention_inputs = attention_structure.apply(
            variables, inputs_embeds, attention_mask, token_type_ids, method=attention_structure.init_attention_inputs
        )
        #
        output, new_attention_inputs = attention_structure.apply(
            variables, inputs_embeds, attention_inputs, method=attention_structure.pre_attention_pooling
        )
        self.assertEqual(output.shape, (2, 2, 768))
        _position_embeds, _token_type_mat, _attention_mask, _cls_mask = attention_inputs
        position_embeds, token_type_mat, attention_mask, cls_mask = new_attention_inputs
        self.assertEqual(position_embeds, _position_embeds)
        self.assertEqual(token_type_mat.shape, (2, 2, 4))
        self.assertEqual(attention_mask.shape, (2, 4))
        self.assertEqual(cls_mask.shape, (2, 4))

    def test_pre_attention_pooling_not_pool_q_only(self):
        batch_size, seq_len = 2, 4
        config = FunnelConfig(pool_q_only=False)
        inputs_embeds  = jax.random.normal(rng, (batch_size, seq_len, config.d_model))
        attention_mask = jnp.array([ [1,1,1,0], [1,1,1,1] ])
        token_type_ids = jnp.array([ [2,1,1,1], [2,2,1,1] ])
        attention_structure = FunnelAttentionStructure(config)
        variables = attention_structure.init(
            rng, inputs_embeds, attention_mask, token_type_ids, method=attention_structure.init_attention_inputs
        )
        attention_inputs = attention_structure.apply(
            variables, inputs_embeds, attention_mask, token_type_ids, method=attention_structure.init_attention_inputs
        )
        #
        output, new_attention_inputs = attention_structure.apply(
            variables, inputs_embeds, attention_inputs, method=attention_structure.pre_attention_pooling
        )
        self.assertEqual(output.shape, (2, 2, 768))
        _position_embeds, _token_type_mat, _attention_mask, _cls_mask = attention_inputs
        position_embeds, token_type_mat, attention_mask, cls_mask = new_attention_inputs
        self.assertEqual(position_embeds, _position_embeds)
        self.assertEqual(token_type_mat.shape, (2, 2, 2))
        self.assertEqual(attention_mask.shape, (2, 2))
        self.assertEqual(cls_mask.shape, (2, 2))

    def test_post_attention_pooling(self):
        batch_size, seq_len = 2, 4
        config = FunnelConfig()
        inputs_embeds  = jax.random.normal(rng, (batch_size, seq_len, config.d_model))
        attention_mask = jnp.array([ [1,1,1,0], [1,1,1,1] ])
        token_type_ids = jnp.array([ [2,1,1,1], [2,2,1,1] ])
        attention_structure = FunnelAttentionStructure(config)
        variables = attention_structure.init(
            rng, inputs_embeds, attention_mask, token_type_ids, method=attention_structure.init_attention_inputs
        )
        attention_inputs = attention_structure.apply(
            variables, inputs_embeds, attention_mask, token_type_ids, method=attention_structure.init_attention_inputs
        )
        output, new_attention_inputs = attention_structure.apply(
            variables, inputs_embeds, attention_inputs, method=attention_structure.pre_attention_pooling
        )
        new_new_attention_inputs = attention_structure.apply(
            variables, new_attention_inputs, method=attention_structure.post_attention_pooling
        )
        position_embeds, token_type_mat, attention_mask, cls_mask = new_new_attention_inputs
        self.assertEqual(token_type_mat.shape, (2, 2, 2))
        self.assertEqual(attention_mask.shape, (2, 2))
        self.assertEqual(cls_mask.shape, (2, 2))
