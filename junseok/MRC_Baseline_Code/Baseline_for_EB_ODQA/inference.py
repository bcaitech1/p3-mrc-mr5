"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.
대부분의 로직은 train.py 와 비슷하나 retrieval, predict
"""

import logging
import os
import sys
from datasets import load_from_disk

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils.hf_inf import 
from arguments import (
    ModelArguments,
    DataTrainingArguments,
    InferencelArguments
)

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(model_args, data_args, inf_args):
    # model_name = model_args.model_name_or_path = './result/monologg_koelectra-base-v3-finetuned-korquad/checkpoint-900'
    model_name = model_args.model_name_or_path
    if model_name == None:
        model_name, model_args.model_name_or_path = get_recent_model(
            './Baseline_for_EB_ODQA/checkpoints')
        # model_name, model_args.model_name_or_path = get_recent_model('./Baseline_for_EB_ODQA/checkpoints')
    else:
        model_name = model_name.replace('/', '_')
    output_dir = f'./submit/{model_name}{model_args.suffix}/'
    logging_dir = f'./logs/{model_name}{model_args.suffix}/'
    training_args = TrainingArguments(
        output_dir=output_dir,          # output directory
        do_predict=True,
        seed=42,
    )
    i = 0
    while os.path.exists(training_args.output_dir):
        training_args.output_dir = f'./submit/{model_name}{model_args.suffix}_{i}/'
        training_args.logging_dir = f'./logs/{model_name}{model_args.suffix}_{i}/'
        i += 1

    print(f"training Data : {training_args}")
    print(f"model Data : {model_args}")
    print(f"data : {data_args}")
    print(f"inference setting : {inf_args}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    if training_args.do_predict:
        data_args.dataset_name = './data/test_dataset'

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    # run passage retrieval if true
    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(datasets, training_args, inf_args)

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args,
                datasets, tokenizer, model)



if __name__ == "__main__":
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    parser = HfArgumentParser(  # hint 만들어주는 것인듯?
        (ModelArguments, DataTrainingArguments, InferencelArguments)
    )
    model_args, data_args, inf_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, inf_args)
