from __future__ import absolute_import, division, print_function
import logging
import os
import random
import sys
from datasets import load_from_disk, DatasetDict

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers.file_utils import HUGGINGFACE_CO_PREFIX

from models.modeling_bert import QuestionAnswering, Config
from utils.optimization import AdamW, WarmupLinearSchedule
from utils.tokenization import BertTokenizer
from utils.korquad_utils import read_squad_examples, convert_examples_to_features
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(train_args, dir_args, token_args, train_dataset, model):
    """ Train the model """
    train_batch_size = train_args.per_device_batch_size // train_args.gradient_accumulation_steps
    train_sampler = RandomSampler(
        train_dataset) if train_args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    t_total = len(
        train_dataloader) // train_args.gradient_accumulation_steps * train_args.epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': train_args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=train_args.learning_rate, eps=train_args.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=t_total*train_args.warmup_ratio, t_total=t_total)
    if train_args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if train_args.fp16_opt_level == "O2":
            keep_batchnorm_fp32 = False
        else:
            keep_batchnorm_fp32 = True
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=train_args.fp16_opt_level, keep_batchnorm_fp32=keep_batchnorm_fp32
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if train_args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, message_size=250000000,
                    gradient_predivide_factor=torch.distributed.get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", train_args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                train_args.per_device_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                train_batch_size
                * train_args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if train_args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                train_args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs = 0
    model.zero_grad()
    model.train()
    train_iterator = trange(int(train_args.epochs),
                            desc="Epoch", disable=train_args.local_rank not in [-1, 0])
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(train_args.seed)
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Train(XX Epoch) Step(X/X) (loss=X.X)", disable=train_args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(device)
                          for t in batch)  # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            outputs = model(input_ids, segment_ids, input_mask,
                            start_positions, end_positions)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs

            if torch.cuda.device_count() > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if train_args.gradient_accumulation_steps > 1:
                loss = loss / train_args.gradient_accumulation_steps

            if train_args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % train_args.gradient_accumulation_steps == 0:
                if train_args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), train_args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_args.max_grad_norm)

                scheduler.step()  # Update learning rate schedule\
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_iterator.set_description(
                    "Train(%d Epoch) Step(%d / %d) (loss=%5.5f)" % (_,
                                                                    global_step, t_total, loss.item())
                )

        if train_args.local_rank in [-1, 0]:
            model_checkpoint = "korquad_{0}_{1}_{2}_{3}.bin".format(train_args.learning_rate,
                                                                    train_batch_size,
                                                                    epochs,
                                                                    int(train_args.epochs))
            logger.info(model_checkpoint)
            output_model_file = os.path.join(dir_args.output_dir, model_checkpoint)
            if torch.cuda.device_count() > 1 or train_args.local_rank != -1:
                logger.info("** ** * Saving file * ** ** (module)")
                torch.save(model.module.state_dict(), output_model_file)
            else:
                logger.info("** ** * Saving file * ** **")
                torch.save(model.state_dict(), output_model_file)
        epochs += 1
    logger.info("Training End!!!")


def load_and_cache_examples(train_args, dir_args, token_args, tokenizer):
    # Load data features from cache or dataset file
    input_file = args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), '_cached_{}_{}_{}'.format('train',
                                                                                               str(
                                                                                                   token_args.max_seq_length),
                                                                                               token_args.doc_stride))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_squad_examples(
            input_file=args.train_file, is_training=True, version_2_with_negative=False)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=token_args.max_seq_length,
                                                doc_stride=token_args.doc_stride,
                                                max_query_length=token_args.max_query_length,
                                                is_training=True)

        if train_args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(features, cached_features_file,
                       pickle_protocol=pickle.HIGHEST_PROTOCOL)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_start_positions = torch.tensor(
        [f.start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor(
        [f.end_position for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask,
                            all_segment_ids, all_start_positions, all_end_positions)
    return dataset


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
    
    if dir_args.huggingface:
        # set output_dir
        if dir_args.get_last_ckpt:
            output_dir = dir_args.model_dir_or_name + "/../"
        else:
            output_dir = dir_args.output_dir + "/" + dir_args.model_dir_or_name.replace('/','_') + dir_args.suffix
            i = 0            
            model_name = dir_args.model_dir_or_name.replace('/','_')
            while os.path.exists(output_dir):
                output_dir = f'{dir_args.output_dir}/{model_name}{dir_args.suffix}_{i}/'
                i += 1
        training_args = TrainingArguments(
            output_dir=output_dir,           # output directory
            save_total_limit=train_args.save_total_limit,              # number of total save model.
            # save_steps=500,                  # model saving step.
            # total number of training epochs
            num_train_epochs=train_args.epochs,
            learning_rate=train_args.learning_rate,
            # batch size per device during training
            per_device_train_batch_size=train_args.per_device_batch_size,
            # batch size for evaluation
            per_device_eval_batch_size=train_args.per_device_batch_size,
            warmup_ratio=train_args.warmup_ratio,
            # warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=train_args.weight_decay,               # strength of weight decay
            # logging_dir=logging_dir,            # directory for storing logs
            # logging_steps=100,              # log saving step.
            evaluation_strategy='steps',  # evaluation strategy to adopt during training
            # `no`: No evaluation during training.
            # `steps`: Evaluate every `eval_steps`.
            # `epoch`: Evaluate every end of epoch.
            # evaluation step.
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
        datasets = load_from_disk(dir_args.data_dir)
        
        if 'validation' not in datasets.column_names:
            datasets = datasets.train_test_split(test_size=0.1)
            datasets = DatasetDict({'train': datasets['train'], 'validation': datasets['test']})

        tokenizer = AutoTokenizer.from_pretrained(
            dir_args.model_dir_or_name,
            use_fast=True,
        )
        config = AutoConfig.from_pretrained(
            dir_args.model_dir_or_name
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
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

        run_mrc(training_args, dir_args, token_args,
                datasets, tokenizer, model)
    else:
        i = 0
        output_dir = f'{dir_args.output_dir}/bert_{"v1" if "v1" in dir_args.model_dir_or_name else"v2"}{dir_args.suffix}/'
        while os.path.exists(output_dir):
            output_dir = f'{dir_args.output_dir}/bert_{"v1" if "v1" in dir_args.model_dir_or_name else"v2"}{dir_args.suffix}_{i}/'
            i += 1
        dir_args.output_dir = output_dir
        # Prepare model
        tokenizer = BertTokenizer(
            dir_args.vocab_dir, max_len=token_args.max_seq_length, do_basic_tokenize=True)
        config = Config.from_json_file(dir_args.config_dir)
        model = QuestionAnswering(config)
        model.bert.load_state_dict(torch.load(dir_args.model_dir_or_name))
        num_params = count_parameters(model)
        logger.info("Total Parameter: %d" % num_params)
        model.to(device)

        # logger.info("Training hyper-parameters %s", args)
        print("Train Arguments :")
        print(train_args)

        print("Directory Arguments:")
        print(dir_args)

        print("Tokenizer Arguments:")
        print(token_args)

        # Training
        train_dataset = load_and_cache_examples(
            train_args, dir_args, token_args, tokenizer)
        train(train_args, dir_args, token_args, train_dataset, model)


if __name__ == "__main__":
    main()
