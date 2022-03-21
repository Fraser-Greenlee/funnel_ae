from typing import Dict
import torch
from torch import nn
from dataclasses import dataclass, field
from transformers.trainer import Trainer, is_torch_tpu_available, get_parameter_names, ShardedDDPOption, is_sagemaker_mp_enabled
from transformers.training_args import TrainingArguments


@dataclass
class AeTrainingArguments(TrainingArguments):
    skip_con_args: str = field(default=None, metadata={"help": "Args for skip connections, one per block, (k, b), 'no' to not use connection. See https://www.desmos.com/calculator/qcl82whtzo"})
    dont_train_encoder: bool = field(default=False, metadata={"help": "Don't optimize encoder parameters."})

    def __post_init__(self):
        super().__post_init__()
        if self.skip_con_args:
            skip_args = []
            for str_args in self.skip_con_args.split(','):
                if str_args == "no":
                    skip_args.append((None, None))
                else:
                    skip_args.append([float(v) for v in str_args.split()])
            self.skip_con_args = skip_args
        else:
            self.skip_con_args = None


class AeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.skip_con_args:
            assert len(self.model.config.block_sizes) == len(self.args.skip_con_args)

    def _skip_connection_weights(self):
        weights = []
        for (k, b) in self.args.skip_con_args:
            if k is None:
                weights.append(0.0)
            else:
                weights.append(
                    torch.sigmoid(
                        torch.tensor(b - self.state.global_step * k)
                    )
                )
        return torch.tensor(weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.skip_con_args and type(self.state.global_step) is int:
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

            if self.args.skip_con_args:
                skip_weights = model.funnel.decoder.skip_w.tolist()
                for i, weight in enumerate(skip_weights):
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

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # allow not training encoder
            model_parameters = [(n, p) for n,p in self.model.named_parameters()]
            if self.args.dont_train_encoder:
                model_parameters = [(n, p) for n,p in model_parameters if 'encoder' not in n]

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model_parameters if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in model_parameters if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                raise NotImplementedError()
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            raise NotImplementedError()

        return self.optimizer
