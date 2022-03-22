from transformers.models.funnel.configuration_funnel import FunnelConfig


class FunnelAeConfig(FunnelConfig):
    def __init__(
        self,
        vocab_size=30522,
        block_sizes=[4, 4, 4],
        block_repeats=None,
        num_decoder_layers=None,
        d_model=768,
        n_head=12,
        d_head=64,
        d_inner=3072,
        hidden_act="gelu_new",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0,
        max_position_embeddings=512,
        type_vocab_size=3,
        initializer_range=0.1,
        initializer_std=None,
        layer_norm_eps=1e-9,
        pooling_type="mean",
        attention_type="relative_shift",
        separate_cls=True,
        truncate_seq=True,
        pool_q_only=True,
        # new
        share_encoder_blocks=False,
        upsample_q_only=False,
        upsample_mode="ff_seperator",
        _randn_enc=False,
        **kwargs
    ):
        self.share_encoder_blocks = share_encoder_blocks
        self.upsample_q_only = upsample_q_only
        self.upsample_mode = upsample_mode
        self._randn_enc = _randn_enc
        super().__init__(vocab_size, block_sizes, block_repeats, num_decoder_layers, d_model, n_head, d_head, d_inner, hidden_act, hidden_dropout, attention_dropout, activation_dropout, max_position_embeddings, type_vocab_size, initializer_range, initializer_std, layer_norm_eps, pooling_type, attention_type, separate_cls, truncate_seq, pool_q_only, **kwargs)
        self.num_decoder_layers = None
