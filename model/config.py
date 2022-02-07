from transformers import FunnelConfig


# TODO mix with FunnelConfig

class FunnelAeConfig(FunnelConfig):
    model_type = "ae"
    is_composition = True

    def __init__(
        self,

        use_attention_upsample=False,
        attention_upsampling_type='bilinear',

        latent_size=32,

        ae_encoder_n_layers=1,
        ae_encoder_use_dropout=False,

        ae_decoder_n_layers=1,
        ae_decoder_use_dropout=False,
        ae_decoder_use_layer_norm=False,
        ae_decoder_layer_norm_eps=1e-09,

        **kwargs
    ):
        super().__init__(**kwargs)

        self.use_attention_upsample = use_attention_upsample
        self.attention_upsampling_type = attention_upsampling_type

        self.latent_size = latent_size

        self.ae_encoder_n_layers = ae_encoder_n_layers
        self.ae_encoder_use_dropout = ae_encoder_use_dropout

        self.ae_decoder_n_layers = ae_decoder_n_layers
        self.ae_decoder_use_dropout = ae_decoder_use_dropout
        self.ae_decoder_use_layer_norm = ae_decoder_use_layer_norm
        self.ae_decoder_layer_norm_eps = ae_decoder_layer_norm_eps
