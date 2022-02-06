from dataclasses import dataclass, field

from transformers.training_args import TrainingArguments


@dataclass
class VaeTrainingArguments(TrainingArguments):
    skip_conn_schedule_k: float =           field(default=0.0025,   metadata={"help": "Multiplied by global_step in a sigmoid, more gradually increase regulariser loss weight."})
    skip_conn_schedule_b: float =           field(default=6.25,     metadata={"help": "Added to global step in sigmoid, further delays increase in regulariser loss weight."})
    skip_conn_schedule_b_offset: float =    field(default=1.0,      metadata={"help": "Added to global step in sigmoid for each earlier layer, delays reduction of early skips."})
