# README
## Description
This is the baseline code for Extraction based ODQA Task.
- Huggingface Trainer based.
- using KLUE dataset.

## install requirements

### windows

```python
pip install -r requirements.txt
```
or
```python
pip3 install -r requirements.txt
```

## Usage

### train.py

```python
python train.py
"""
Arguments for Training Model.
"""
--k_fold: int "Not Implemented! model number of k-fold validation(default: no)"
--learning_rate: float "The initial learning rate for Adam. (default: 5e-5)"
--epochs: int "Total number of training epochs to perform.(default: 4)"
--per_device_batch_size: int "Total batch size for training. (default: 16)"
--weight_decay: float "weight decay (default: 0.01)"
--adam_epsilon: float "Epsilon for Adam optimizer. (default: 1e-6)"
--warmup_ratio: float "ratio of training to perform linear learning rate warmup for. E.g., 0.1 = 10\% of training. (default: 0.1)
--fp16: bool "Whether to use 16-bit (mixed) precision training instead of 32-bit training (default: True)
--fp16_opt_level: str "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. (default: O2) See details at https://nvidia.github.io/apex/amp.html"
--save_total_limit: int "number of total save model.(default : 1)"
--evaluation_step_ratio: int "step term about evaluation and save model / per_device_batch_size(default: 16)"
--gradient_accumulation_steps: Optional[int] "Number of updates steps to accumulate before performing a backward/update pass. (default: 1)"
--max_grad_norm: Optional[float] "Maximum gradient norm (default: 1.0)"
--seed: Optional[int] "random seed (default: 1)"
--local_rank: Optional[int] "For distributed training: local_rank (default: -1)"
    
"""
Arguments for Tokenizer Settings
"""
--max_seq_length: int"The maximum total input sequence length after tokenization. (default: 512)"
--doc_stride: int "When splitting up a long document into chunks, how much stride to take between chunks. (default: 128)"
--pad_to_max_length: bool "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch (which can be faster on GPU but will be slower on TPU)."
--max_query_length: int "The maximum number of tokens for the question. Questions longer than this will be truncated to this length. (default: 64)"
--max_answer_length: int "The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another. (default : 30)"
    
"""
Arguments for Data and Output target directoies.
"""
--output_dir: str "directory to save checkpoints(default: ./checkpoints/\{model_dir_or_name\}/)"
--model_dir_or_name: str "directory to get model or name of model in huggingface/model (default: monologg/koelectra-base-v3-finetuned-korquad)"
--suffix: str "suffix for model distinction. (default: None)"
--data_dir: str "directory to get training data. (default: ./data/korquad/)"
--submit_dir: str "directory to predcition output for submission. (default: ./submit/\{model_dir_or_name\})"
--config_dir: Optional[str] "directory to model configuration file(json) (default: using AutoConfig in huggingface/transformers)"
--vocab_dir: Optional[str] "directory to tokenizer vacab file(json) (default: using AutoTokenizer in huggingface/transformers)"

```

### inference.py


```python
python inference.py
"""
Arguments for Trainer.
"""
--seed: Optional[int] "random seed (default: 1)"
    
"""
Arguments for Data and Output target directoies.
"""
--output_dir: str "directory to save checkpoints(default: ./checkpoints/\{model_dir_or_name\}/)"
--model_dir_or_name: str "directory to get model or name of model in huggingface/model (default: monologg/koelectra-base-v3-finetuned-korquad)"
--suffix: str "suffix for model distinction. (default: None)"
--data_dir: str "directory to get training data. (default: ./data/korquad/)"
--submit_dir: str "directory to predcition output for submission. (default: ./submit/\{model_dir_or_name\})"
--config_dir: Optional[str] "directory to model configuration file(json) (default: using AutoConfig in huggingface/transformers)"
    
"""
Arguments Inference and Evaluation step.
"""
retri: str "retrieval method for retrieval step(default: bm25plus)"
k: int "top-k number(default: 5)"
top_join : str "Not Implemented! top-k merge method('sep': evaluate each k, 'whole': merge to one article)(default: 'whole')"
is_eval: bool "Not Implemented! check True if you want to use validation dataset."    
```

