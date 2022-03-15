from typing import List, Optional, Tuple

import torch
from dataclasses import dataclass
from transformers.file_utils import ModelOutput

from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput


@dataclass
class AutoEncOutput(Seq2SeqLMOutput):
    pass


@dataclass
class BaseVaeOutput(ModelOutput):
    """
    Base class for an VAE's outputs.

    Args:
        reconstructed_encoding (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Reconstructed hidden states originally from the last layer of the encoder.
        latent (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, latent_size)`):
            Latent codes representing encoded sequences.
        reg_loss (:obj:`torch.FloatTensor` of shape :obj:`(batch_size)`):
            MMD-VAE regularisation loss for this step.
    """

    latent: torch.FloatTensor = None
    reconstructed_encoding: torch.FloatTensor = None
    reg_loss: Optional[torch.FloatTensor] = None


@dataclass
class TrackAttentionInputsOutput(BaseModelOutput):
    all_attention_inputs: List[Tuple[torch.tensor]] = None
