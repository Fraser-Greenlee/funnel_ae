from typing import Dict
import torch
from dataclasses import dataclass, field
from transformers.trainer import Trainer, is_torch_tpu_available
from transformers.training_args import TrainingArguments


@dataclass
class AeTrainingArguments(TrainingArguments):
    use_skip_con: bool = field(default=True,  metadata={"help": "Use skip connection weights."})
    skip_conn_schedule_k: float = field(default=0.0014,   metadata={"help": "Multiplied by global_step in a sigmoid, more gradually increase regulariser loss weight."})
    skip_conn_schedule_b: float = field(default=3,     metadata={"help": "Added to global step in sigmoid, further delays increase in regulariser loss weight."})
    skip_conn_schedule_b_offsets: str = field(default="", metadata={"help": "Delays reduction of early skip layers."})

    def __post_init__(self):
        super().__post_init__()
        self.skip_conn_schedule_b_offsets = [
            float(v) for v in self.skip_conn_schedule_b_offsets.split()
        ]

class AeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.skip_conn_schedule_b_offsets:
            assert len(self.model.config.block_sizes) == len(self.args.skip_conn_schedule_b_offsets)
        else:
            self.args.skip_conn_schedule_b_offsets = [0 for _ in self.model.config.block_sizes]
        self.skip_conn_schedule_b_offsets = torch.tensor(self.args.skip_conn_schedule_b_offsets, requires_grad=False)

    def _skip_connection_weights(self):
        return torch.sigmoid(
            self.state.global_step * self.args.skip_conn_schedule_k
            - self.args.skip_conn_schedule_b
            -  self.skip_conn_schedule_b_offsets
        ).tolist()

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.use_skip_con and self.state.global_step:
            model.funnel.decoder.skip_w = self._skip_connection_weights()
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        '''
            Adds logging `skip_con_{i}`
        '''
        if self.control.should_log:
            if is_torch_tpu_available():
                raise NotImplementedError()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            if self.args.use_skip_con:
                for i, weight in enumerate(model.funnel.decoder.skip_w):
                    logs[f"skip_con_{i}"] = weight

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
