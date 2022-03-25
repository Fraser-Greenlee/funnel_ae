import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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

    def _enc_dataset(self, model, dataloader):

        logger.info(f"***** Running Encoding *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {dataloader.batch_size}")

        input_ids, logits, hidden_states = None, None, None
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in tqdm(enumerate(dataloader), desc="Encode test dataset."):
            if step > 5:
                break
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

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

        latents = hidden_states[1 + (torch.tensor(model.config.block_sizes) * torch.tensor(model.config.block_repeats)).sum()]
        latent_tokens = latents.view(-1, latents.shape[-1])

        return input_ids, logits, hidden_states, latent_tokens

    def plot_pca(self, hidden_states):
        hidden_exp_variances = []
        latent_exp_variances = []
        d_model = self.model.config.d_model
        d_latent = self.model.config.d_latent
        for hidden_layer in hidden_states:
            pca = PCA()
            pca.fit_transform(hidden_layer.view(-1, hidden_layer.shape[-1]))
            pca.explained_variance_.sort()
            exp_variance = pca.explained_variance_[::-1].tolist()
            if hidden_layer.shape[-1] == d_model:
                hidden_exp_variances.append(exp_variance)
            elif hidden_layer.shape[-1] == d_latent:
                latent_exp_variances.append(exp_variance)
            else:
                raise Exception('Bad hidden shape.')

        wandb.log({"PCA component explained variance per hidden token." : wandb.plot.line_series(
                xs=range(d_model),
                ys=hidden_exp_variances,
                keys=[f'layer_{i}' for i in range(len(hidden_exp_variances))],
                title="PCA component explained variance per hidden token.",
                xname="component")})

        wandb.log({"PCA component explained variance per latent token." : wandb.plot.line_series(
                xs=range(d_latent),
                ys=latent_exp_variances,
                keys=[f'latent_{i}' for i in range(len(latent_exp_variances))],
                title="PCA component explained variance per latent token.",
                xname="component")})

    def plot_latents(self, latent_tokens):
        # expect gaussian distributions across samples and dimensions
        to_log = {}
        for perplexity in tqdm([2, 5, 30, 50, 100], desc='t-sne'):
            tsne = TSNE(perplexity=perplexity, random_state=123, init='pca', learning_rate='auto')
            z = tsne.fit_transform(latent_tokens)
            table = wandb.Table(data=z, columns = ["z_0", "z_1"])
            # TODO log row `label` with scatter values, should see unsupervised classification
            to_log[f"latent t-sne {perplexity} perplexity"] = wandb.plot.scatter(table, "z_0", "z_1", title=f"T-SNE latent token {perplexity} perplexity.")
        wandb.log(to_log)

        lt_min, lt_max = latent_tokens.min(), latent_tokens.max()
        lt_max = (lt_max + (1 if lt_max.item() > 0 else -1)).int().item()
        lt_min = (lt_min + (1 if lt_min.item() > 0 else -1)).int().item()
        bins = torch.linspace(lt_min, lt_max, (abs(lt_max) + abs(lt_min)) * 10+1)

        def _hist(hidden):
            hist = torch.histogram(hidden, bins).hist
            return hist/hist.sum()

        latent_hist = _hist(latent_tokens)
        ideal_hist = _hist(torch.randn_like(latent_tokens))

        wandb.log({"Latent values." : wandb.plot.line_series(
                xs=bins,
                ys=[latent_hist, ideal_hist],
                keys=['z', 'ideal'],
                title="Latent token sample values.",
                xname="latent value")})

        dims = []
        for z_i in range(latent_tokens.shape[-1]):
            z_hist = _hist(latent_tokens[:,z_i])
            delta = (z_hist - latent_hist).abs().sum()
            dims.append((delta.item(), z_i, z_hist))
        worst_dims = sorted(dims)[::-1][:10]

        wandb.log({"Latent token dimension values." : wandb.plot.line_series(
                xs=bins,
                ys=[latent_hist] + [
                    z_hist for (_delta, z_i, z_hist) in worst_dims
                ],
                keys=['z_total'] + [f'z_{z_i}' for (_delta, z_i, z_hist) in worst_dims],
                title="Latent token dimension values (+ 10 most unusual).",
                xname="latent value")})

    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:

        dataloader = self.get_eval_dataloader(eval_dataset)
        model = self._wrap_model(self.model, training=False)
        model.eval()

        input_ids, logits, hidden_states, latent_tokens = self._enc_dataset(model, dataloader)

        preds = self.tokenizer.batch_decode(logits.max(dim=2).indices)
        targets = self.tokenizer.batch_decode(input_ids)
        table = wandb.Table(columns=['prediction', 'target'], data=list(zip(preds, targets)))
        wandb.log({'Predicted text': table})

        # TODO have model learn to predict latent feature based on surrounding features.
        # Have a seperate linear layer to predict each latent feature
        # Would be great to be able to counter for consistancies in the dataset?
        # Could have an embedding -> pred model to counter for dataset distribution?

        self.plot_pca(hidden_states)
        self.plot_latents(latent_tokens)

        breakpoint()

        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
