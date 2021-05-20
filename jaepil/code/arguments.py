from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import (
    TrainingArguments,
)
from transformers.trainer_utils import IntervalStrategy

from pathlib import Path
import sys

# BASE_PATH = Path('.').resolve().parent
BASE_PATH = Path("/opt/ml/jaepil").resolve()
print(f"Current BASE_PATH is: {BASE_PATH}")
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
        default=data_path.default / "kluquad",
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
class MyTrainingArguments(TrainingArguments):
    """Inherits transformers.TrainingArguments to manage configs here
    """    
    output_dir: Union[str, Path] = field(
        default=PathArguments.model_output_path,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
        )
    num_train_epochs: float = field(
        default=3.0, 
        metadata={"help": "Total number of training epochs to perform."},
        )
    learning_rate: float = field(
        default=1e-5, 
        metadata={"help": "The initial learning rate for AdamW."}
        )
    warmup_steps: int = field(
        default=500, 
        metadata={"help": "Linear warmup over warmup_steps."}
        )
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    load_best_model_at_end: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )
    metric_for_best_model: Optional[str] = field(
        default="exact_match", metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=True, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )

    # save steps ignored if load_best_model_at_end is True
    save_steps: int = field(
        default=1000,
        metadata={"help": "Save checkpoint every X updates steps."},
        )
    save_total_limit: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    save_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
        # save_strategy (:obj:`str` or :class:`~transformers.trainer_utils.IntervalStrategy`, `optional`, defaults to :obj:`"steps"`):
        #     The checkpoint save strategy to adopt during training. Possible values are:

        #         * :obj:`"no"`: No save is done during training.
        #         * :obj:`"epoch"`: Save is done at the end of each epoch.
        #         * :obj:`"steps"`: Save is done every :obj:`save_steps`.

@dataclass
class MyInferenceArguments(TrainingArguments):
    """Inherits transformers.TrainingArguments to manage configs here
    """    
    output_dir: Union[str, Path] = field(
        default=PathArguments.test_output_path,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
        )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    k: int = field(
        default=20,
        metadata={"help": "Top k passage to retrieve"}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="monologg/koelectra-base-v3-finetuned-korquad",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )

@dataclass
class RetrievalArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="bert-base-multilingual-cased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    pororo_tokenizer_name: Optional[str] = field(
        default="mecab.bpe64k.ko",
        metadata={"help": """
        Pororo's Korean tokenizers. Select from:
        
        bpe4k.ko, 
        bpe8k.ko, 
        bpe16k.ko, 
        bpe32k.ko, 
        bpe64k.ko, 

        unigram4k.ko, 
        unigram8k.ko, 
        unigram16k.ko, 
        unigram32k.ko, 
        unigram64k.ko, 

        jpe4k.ko, 
        jpe8k.ko, 
        jpe16k.ko, 
        jpe32k.ko, 
        jpe64k.ko, 

        mecab.bpe4k.ko, 
        mecab.bpe8k.ko, 
        mecab.bpe16k.ko, 
        mecab.bpe32k.ko, 
        mecab.bpe64k.ko, 
        
        char, 
        jamo, 
        word, 
        mecab_ko, 
        sent_ko
        """},
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




