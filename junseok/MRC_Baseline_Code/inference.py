"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.
대부분의 로직은 train.py 와 비슷하나 retrieval, predict
"""

import logging
import os
import sys
from datasets import load_metric, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict, DatasetBuilder

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import postprocess_qa_predictions, check_no_error, tokenize
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval, SparseRetrieval_BM25, SparseRetrieval_BM25PLUS

from arguments import (
    ModelArguments,
    DataTrainingArguments,
    InferencelArguments
)

import json
import pandas as pd

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_recent_model(models_dir='./result'):
    all_models = [
        models_dir+'/'+d for d in os.listdir(models_dir) if os.path.isdir(models_dir+'/'+d)]
    latest_models = max(all_models, key=os.path.getmtime)
    all_checkpoints = [latest_models+'/'+d for d in os.listdir(
        latest_models) if os.path.isdir(latest_models+'/'+d)]
    latest_checkpoints = max(all_checkpoints, key=os.path.getmtime)
    return latest_models.replace(models_dir+'/', ''), latest_checkpoints


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
    # if "ko" in model_args.model_name_or_path and "bert" in model_args.model_name_or_path:
    #     print(f"using korean tokenizer for {model_name}")
    #     tokenizer = KoBertTokenizer.from_pretrained(
    #         model_args.tokenizer_name
    #         if model_args.tokenizer_name
    #         else model_args.model_name_or_path,
    #         use_fast=True
    #     )
    # else:
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


def run_sparse_retrieval(datasets, training_args, inf_args):
    #### retreival process ####
    if inf_args.retrieval == None:
        # retriever = SparseRetrieval(tokenize_fn=tokenize,
        #                             data_path="./data",
        #                             context_path="wikipedia_documents.json")
        retriever = SparseRetrieval_BM25PLUS(tokenize_fn=tokenize,
                                             data_path="./data",
                                             context_path="wikipedia_documents.json")
    elif inf_args.retrieval.lower() == "bm25plus" or inf_args.retrieval.lower() == "bm25p":
        retriever = SparseRetrieval_BM25PLUS(tokenize_fn=tokenize,
                                             data_path="./data",
                                             context_path="wikipedia_documents.json")
    elif inf_args.retrieval.lower() == "bm25":
        retriever = SparseRetrieval_BM25(tokenize_fn=tokenize,
                                         data_path="./data",
                                         context_path="wikipedia_documents.json")
    retriever.get_sparse_embedding()
    dfs = retriever.retrieve(
        datasets['validation'].select(range(11)), inf_args.k)

    # faiss retrieval
    # df = retriever.retrieve_faiss(dataset['validation'])
    dfs= list(map(pd.DataFrame, dfs))
    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features({
            'score': Value(dtype='float', id=None),
            'context': Value(dtype='string', id=None),
            'id': Value(dtype='string', id=None),
            'question': Value(dtype='string', id=None)})

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features({'answers': Sequence(feature={'text': Value(dtype='string', id=None),
                                                   'answer_start': Value(dtype='int32', id=None)},
                                          length=-1, id=None),
                      'context': Value(dtype='string', id=None),
                      'id': Value(dtype='string', id=None),
                      'question': Value(dtype='string', id=None)})
    dataset_topk = []
    for df in dfs:
        dataset_topk.append(DatasetDict(
            {'validation': Dataset.from_pandas(df, features=f)}))
    return dataset_topk


def run_mrc(data_args, training_args, model_args, datasets, tokenizer, model):
    # only for eval or predict
    column_names = datasets[0]["validation"].column_names
    # print(datasets['validation']['contexts'])
    data_num = datasets[0]["validation"].num_rows
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    # check if there is an error
    last_checkpoint, max_seq_length = check_no_error(
        training_args, data_args, tokenizer, datasets[0])
    # last_checkpoint, max_seq_length = None, min(data_args.max_seq_length, tokenizer.model_max_length)

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        # with open('./data/kor_stops.json') as jf:
        #     kor_stop = json.load(jf)

        # for i in range(len(examples['contexts'])):
        #     examples['contexts'][i] = examples['contexts'][i].replace(
        #         "\n\n", " ")
        #     for stop in kor_stop['stop_words']:
        #         examples['contexts'][i] = examples['contexts'][i].replace(
        #             f" {stop} ", " ")

        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(
                examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples
    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data collator.
    data_collator = (
        DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )

    def prepare_validation_feature_mapping(dataset):
        eval_dataset = dataset["validation"]

        # Validation Feature Creation
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        return eval_dataset
    eval_datasets = list(map(prepare_validation_feature_mapping, datasets))

    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        all_predict, predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        if training_args.do_predict:
            return all_predict, formatted_predictions, 

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Initialize our Trainer
    logger.info("*** Evaluate ***")
    top_predictions = []
    for eval_idx, eval_dataset in enumerate(eval_datasets):
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=eval_dataset,
            eval_examples=datasets[eval_idx]['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=post_processing_function,
            compute_metrics=compute_metrics,
        )


        # eval dataset & eval example - will create predictions.json
        if training_args.do_predict:
            all_predict, predictions  = trainer.predict(test_dataset=eval_dataset,
                                        test_examples=datasets[eval_idx]['validation'])
            top_predictions.append(all_predict)
        # predictions.json is already saved when we call postprocess_qa_predictions(). so there is no need to further use predictions.
    top_predictions = pd.DataFrame(top_predictions)
    results = []
    
    for row_i in range(data_num):   
        for rank_i in range(inf_args.k): 
            top_predictions[row_i][rank_i].pop('start_logit')
            top_predictions[row_i][rank_i].pop('end_logit')
            top_predictions[row_i][rank_i]['weighted_prob'] = top_predictions[row_i][rank_i]['probability'] * (datasets[rank_i]['validation']['score'][row_i])


    print("No metric can be presented because there is no correct answer given. Job done!")


if __name__ == "__main__":
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    parser = HfArgumentParser(  # hint 만들어주는 것인듯?
        (ModelArguments, DataTrainingArguments, InferencelArguments)
    )
    model_args, data_args, inf_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, inf_args)
