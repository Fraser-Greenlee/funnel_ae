from dataclasses import dataclass
from typing import Dict
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


@dataclass
class AeTrainingArguments(TrainingArguments):
    pass

class AeTrainer(Trainer):

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            logs['masked_lm_loss'] = round(model.get_masked_lm_loss() / (self.state.global_step - self._globalstep_last_logged), 4)
            logs['reg_loss'] = round(model.get_reg_loss() / (self.state.global_step - self._globalstep_last_logged), 4)

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

