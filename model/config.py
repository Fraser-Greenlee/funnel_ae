from transformers import FunnelConfig


class FunnelAeConfig(FunnelConfig):
    model_type = "ae"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.separate_cls = False
        self.upsample_q_only = False
        self.pool_q_only = False
        self.decoder_block_sizes = self.block_sizes[::-1]
        self.pad_token_id = 0
