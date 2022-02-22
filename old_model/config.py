from transformers import FunnelConfig


class FunnelAeConfig(FunnelConfig):
    model_type = "ae"
    is_composition = True

    def __init__(
        self,

        upsampling_type='bilinear',
        upsample_q_only=True,

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

        self.decoder_block_sizes = self.block_sizes[::-1]

        self.upsampling_type = upsampling_type
        self.upsample_q_only = upsample_q_only

        self.latent_size = latent_size

        self.ae_encoder_n_layers = ae_encoder_n_layers
        self.ae_encoder_use_dropout = ae_encoder_use_dropout

        self.ae_decoder_n_layers = ae_decoder_n_layers
        self.ae_decoder_use_dropout = ae_decoder_use_dropout
        self.ae_decoder_use_layer_norm = ae_decoder_use_layer_norm
        self.ae_decoder_layer_norm_eps = ae_decoder_layer_norm_eps
