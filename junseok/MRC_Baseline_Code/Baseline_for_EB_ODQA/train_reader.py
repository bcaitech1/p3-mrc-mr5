from __future__ import absolute_import, division, print_function
import logging
import os
import sys
from datasets import load_from_disk, DatasetDict, load_dataset
import pandas as pd
import torch
from utils.hf_train import (
    run_mrc
)

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from utils.arguments import (
    TrainingArgumentsInputs,
    DirectoryArgumentsInputs,
    TokenizerArgumentsInputs
)

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (TrainingArgumentsInputs, DirectoryArgumentsInputs, TokenizerArgumentsInputs)
    )
    train_args, dir_args, token_args = parser.parse_args_into_dataclasses()

    # Setup CUDA, GPU & distributed training
    if train_args.local_rank == -1:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(train_args.local_rank)
        device = torch.device("cuda", train_args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        # n_gpu = 1

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if train_args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
                   train_args.local_rank, device, bool(train_args.local_rank != -1), train_args.fp16)

    # Set seed
    set_seed(train_args.seed)

    # set output_dir
    output_dir = dir_args.output_dir + "/" + \
        dir_args.model_dir_or_name.replace('/', '_') + dir_args.suffix
    i = 0
    model_name = dir_args.model_dir_or_name.replace('/', '_')
    while os.path.exists(output_dir):
        output_dir = f'{dir_args.output_dir}/{model_name}{dir_args.suffix}_{i}/'
        i += 1
    training_args = TrainingArguments(
        output_dir=output_dir,
        save_total_limit=train_args.save_total_limit,
        # total number of training epochs
        num_train_epochs=train_args.epochs,
        learning_rate=train_args.learning_rate,
        per_device_train_batch_size=train_args.per_device_batch_size,
        per_device_eval_batch_size=train_args.per_device_batch_size,
        warmup_ratio=train_args.warmup_ratio,
        weight_decay=train_args.weight_decay,               # strength of weight decay
        evaluation_strategy='steps',  # evaluation strategy to adopt during training
        adam_epsilon=train_args.adam_epsilon,
        eval_steps=train_args.evaluation_step_ratio * train_args.per_device_batch_size,
        dataloader_num_workers=4,
        load_best_model_at_end=True,  # save_strategy, save_steps will be ignored
        metric_for_best_model="exact_match",  # eval_accuracy
        greater_is_better=True,  # set True if metric isn't loss
        label_smoothing_factor=0.5,
        fp16=train_args.fp16,
        fp16_opt_level=train_args.fp16_opt_level,
        do_train=True,
        do_eval=True,
        seed=train_args.seed,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        max_grad_norm=train_args.max_grad_norm,
        local_rank=train_args.local_rank
    )
    if dir_args.data_dir == "korquad":
        datasets = load_dataset('squad_kor_v1')
    else:
        datasets = load_from_disk(dir_args.data_dir)
    dataset_list = []
    if train_args.k_fold > 1:
        dataset_len = len(datasets)
        for i in range(train_args.k_fold):
            dataset = datasets.select(range(int(dataset_len*(i/train_args.k_fold)), int(dataset_len*(i/train_args.k_fold))))
            dataset=dataset.train_test_split(test_size=1/train_args.k_fold)
            dataset=DatasetDict(
                {'train': dataset['train'], 'validation': dataset['test']})
            dataset_list.append(dataset)
    elif 'validation' not in datasets.column_names:
        datasets=datasets.train_test_split(test_size=0.1)
        datasets=DatasetDict(
            {'train': datasets['train'], 'validation': datasets['test']})
        dataset_list.append(datasets)
    else:
        dataset_list.append(datasets)

    config=AutoConfig.from_pretrained(
        dir_args.config_dir
        if dir_args.config_dir
        else dir_args.model_name_or_path,
    )
    tokenizer=AutoTokenizer.from_pretrained(
        dir_args.vocab_dir
        if dir_args.vocab_dir
        else dir_args.model_name_or_path,
        use_fast=True,
    )

    model=AutoModelForQuestionAnswering.from_pretrained(
        dir_args.model_dir_or_name,
        from_tf=bool(".ckpt" in dir_args.model_dir_or_name),
        config=config,
    )

    print("Train Arguments :")
    print(training_args)

    print("Directory Arguments:")
    print(dir_args)

    print("Tokenizer Arguments:")
    print(token_args)
    root_dir=output_dir
    for idx, dataset in enumerate(dataset_list):
        training_args.output_dir=root_dir+f'/{idx}'
        run_mrc(training_args, dir_args, token_args,
                dataset, tokenizer, model)


if __name__ == "__main__":
    main()
