from dataclasses import dataclass
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


@dataclass
class AeTrainingArguments(TrainingArguments):
    pass

class AeTrainer(Trainer):
    pass
