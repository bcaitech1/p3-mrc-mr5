from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import (
    TrainingArguments,
)

from pathlib import Path
import sys

BASE_PATH = Path('.').resolve().parent
sys.path.append(BASE_PATH.as_posix())
@dataclass
class PathArguments:
    """Arguments that point to various data input & output paths
    """    
    data_path: Union[str, Path] = field(
        default=BASE_PATH / "input" / "data" / "data",
        metadata={"help": "Actual root path of the data"},
        )
    train_path: Union[str, Path] = field(
        default=data_path.default / "train_dataset",
        metadata={"help": "Wrapper train data path containing metadata"},
        )
    train_data_path: Union[str, Path] = field(
        default=train_path.default / "train",
        metadata={"help": "Actual train data path containing pyarrow format data"}
        )
    val_data_path: Union[str, Path] = field(
        default=train_path.default / "validation",
        metadata={"help": "Actual validation(train) data path containing pyarrow format data"},
        )
    test_path: Union[str, Path] = field(
        default=data_path.default / "test_dataset",
        metadata={"help": "Wrapper test data path containing metadata"},
        )
    test_data_path: Union[str, Path] = field(
        default=test_path.default / "validation",
        metadata={"help": "Actual validation(test) data path containing pyarrow format data"},
        )
    
    output_path: Union[str, Path] = field(
        default=BASE_PATH / "output",
        metadata={"help": "Wrapper output data path containing train/test/models/processed directories"},
    )
    processed_path: Union[str, Path] = field(
        default=output_path.default / "processed_data",
        metadata={"help": "Temp directory to store processed data"},
    )
    train_output_path: Union[str, Path] = field(
        default=output_path.default / "train_dataset",
        metadata={"help": "Actual output(train) data path containing inference result"},
    )
    test_output_path: Union[str, Path] = field(
        default=output_path.default / "test_dataset",
        metadata={"help": "Actual output(test) data path containing inference result"},
    )
    model_output_path: Union[str, Path] = field(
        default=output_path.default / "models" / "train_dataset",
        metadata={"help": "Model output path containing trained models"},
    )

@dataclass
class TrainingArguments(TrainingArguments):
    """Inherits transformers.TrainingArguments to manage configs here
    """    
    output_dir: Union[str, Path] = field(
        default=PathArguments.model_output_path,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="bert-base-multilingual-cased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=PathArguments.train_path, metadata={"help": "The name of the dataset to use."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
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




