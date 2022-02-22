from transformers import FunnelConfig


class FunnelAeConfig(FunnelConfig):
    model_type = "ae"

    def __init__(
        self,
        upsample_q_only=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.upsample_q_only = upsample_q_only
        self.decoder_block_sizes = self.block_sizes[::-1]
