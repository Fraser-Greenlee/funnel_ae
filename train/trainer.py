import torch
from torch.nn import functional as F
from transformers.trainer import Trainer


class AeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_pools = len(self.model.config.block_sizes)

    def skip_connection_weights(self):
        if self.state.global_step is None:
            return 0

        base_weight = self.state.global_step * self.args.skip_conn_schedule_k - self.args.skip_conn_schedule_b
        skip_conn_offsets = torch.arange(0, self.self.n_pools) * self.args.skip_conn_schedule_b_offset

        return torch.sigmoid(base_weight - skip_conn_offsets).item()

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        model.skip_connection_wieghts = self.skip_connection_weights()
        outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if not self.args.no_reg_loss:
            reg_loss = self.reg_loss(outputs.latent)
            loss += self.reg_weight() * reg_loss

        if self.args.recon_loss:
            loss += F.mse_loss(outputs["reconstructed_encoding"], outputs["encoder_last_hidden_state"])

        return (loss, outputs) if return_outputs else loss
