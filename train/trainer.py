import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers.trainer import Trainer
from transformers.trainer_utils import has_length
from transformers.trainer_pt_utils import find_batch_size
from transformers.training_args import TrainingArguments
from transformers.utils import logging


logger = logging.get_logger(__name__)


@dataclass
class AeTrainingArguments(TrainingArguments):
    def __post_init__(self):
        if self.output_dir and not torch.cuda.is_available():
            self.output_dir = 'test'
        return super().__post_init__()


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

    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:

        dataloader = self.get_eval_dataloader(eval_dataset)
        batch_size = dataloader.batch_size

        logger.info(f"***** Running Encoding *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model = self._wrap_model(self.model, training=False)
        model.eval()

        logits, hidden_states = None, None
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in tqdm(enumerate(dataloader), desc="Encode test dataset."):
            if step > 5:
                break
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            inputs.update(dict(
                output_attentions=False, # TODO could probably use the attention pattern
                output_hidden_states=True,
                return_dict=True,
            ))
            outputs = model(**inputs)

            if logits is None:
                logits = outputs.logits.detach().cpu()
            else:
                logits = torch.concat((logits, outputs.logits.detach().cpu()))

            if input_ids is None:
                input_ids = inputs['input_ids'].cpu()
            else:
                input_ids = torch.concat((input_ids, inputs['input_ids']))

            if hidden_states is None:
                hidden_states = [v.detach().cpu() for v in outputs.hidden_states]
            else:
                hidden_states = [
                    torch.concat((v, outputs.hidden_states[i].detach().cpu())) for i, v in enumerate(hidden_states)
                ]

        # TODO
        # - log logits greedy VS true predictions
        # - plot histogram values across all latent dimensions
        # - plot pca of hidden dimensions
        # - plot t-SNE of latent codes across test dataset for several ?
        preds = self.tokenizer.batch_decode(logits.max(dim=2).indices)
        targets = self.tokenizer.batch_decode(input_ids)
        latents = hidden_states[1 + (torch.tensor(model.config.block_sizes) * torch.tensor(model.config.block_repeats)).sum()]

        # TODO see how to do wandb plots
        breakpoint()

        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
