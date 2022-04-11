import unittest

import jax

from transformers import FunnelConfig, FunnelTokenizer

from model.modeling_funnel_flax import (
    FunnelEmbeddings,
    FunnelAttentionStructure
)


tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/small")
input_ids = tokenizer("Hello there", return_tensors="np").input_ids
rng = jax.random.PRNGKey(0)


class Basics(unittest.TestCase):
    def test_embeddings_run_default(self):
        config = FunnelConfig()
        emb = FunnelEmbeddings(config)
        variables = emb.init(rng, input_ids)
        embeddings = emb.apply(variables, input_ids)
        self.assertEqual(embeddings.shape, (1, input_ids.shape[-1], config.d_model))

    def test_attn_struct_run_default(self):
        config = FunnelConfig()
        attention_structure = FunnelAttentionStructure(config)
