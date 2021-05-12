from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class InferencelArguments:
    """
    Arguments pertaining to which retrieval Model we are going to inference.
    """
    retrieval: str = field(
        default=None,
        metadata={"help": "retrieval method for retrieval step(default: TF-IDF Sparse)"}
    )
@dataclass
class TrainArguments:
    learning_rate: float = field(
        default=5e-5
    )
    num_train_epochs: int = field(
        default=4
    )
    per_device_train_batch_size: int = field(
        default=32
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    suffix: str = field(
        default="",
        metadata={"help": "suffix name for log, result, submit data folder(default: None)"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    reservation: str = field(
        default=None,
        metadata={"help": "Path to Training Arguments from outside and Training Schedule. If you set this path, all other given arguments will be ignored. (default: None)"}
    )
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default="./data/KLUQUAD", metadata={"help": "The name of the dataset to use."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
                    "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        },
    )
    train_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to train sparse/dense embedding (prepare for retrieval)."},
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help":"Whether to run passage retrieval using sparse/dense embedding )."},
    )