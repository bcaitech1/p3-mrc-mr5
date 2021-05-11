from arguments import InferencelArguments, ModelArguments, DataTrainingArguments
from transformers import TrainingArguments, HfArgumentParser

import os
import json

def get_recent_reservation(reservation_dir='./reservation/'):
    all_res = [reservation_dir+'/'+d for d in os.listdir(reservation_dir) if not os.path.isdir(reservation_dir+'/'+d)]
    latest_res = max(all_res, key=os.path.getmtime)
    return latest_res

def json_to_dataclass(jsondata):
    data_args, model_args, inf_args = None, None, None
    for k, v in jsondata.items():
        if k=="data":
            data_args = DataTrainingArguments()
            for arg_name, arg_val in v.items():
                setattr(data_args, arg_name, arg_val)
        elif k=="model":
            model_args = ModelArguments()
            for arg_name, arg_val in v.items():
                setattr(model_args, arg_name, arg_val)
        elif k=="inference":
            inf_args = InferencelArguments()
            for arg_name, arg_val in v.items():
                setattr(inf_args, arg_name, arg_val)
    return data_args, model_args, inf_args

def get_reservation(path=None):
    if not path:
        path = get_recent_reservation()

    with open(path) as json_file:
        reservation_json = json.load(json_file)

    joblist = []
    for job in reservation_json['jobs']:
        data_args, model_args, inf_args= json_to_dataclass(job)
        training_args = job['train']
        temp = model_args.model_name_or_path
        if temp==None:
            temp = model_args.model_name_or_path = "xlm-roberta-base"
        temp = temp.replace('/','_')
        output_dir= f'./result/{temp}{model_args.suffix}/'
        logging_dir= f'./logs/{temp}{model_args.suffix}/'

        training_args = TrainingArguments(
            output_dir=output_dir,           # output directory
            save_total_limit=training_args['save_total_limit'] or 2,              # number of total save model.
            save_steps=training_args['save_steps'] or 500,                  # model saving step.
            num_train_epochs=training_args['num_train_epochs'] or 5,              # total number of training epochs
            learning_rate=training_args['learning_rate'] or 5e-5,              # learning_rate
            per_device_train_batch_size=32,  # batch size per device during training
            per_device_eval_batch_size=32,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir=logging_dir,            # directory for storing logs
            logging_steps=100,              # log saving step.
            evaluation_strategy='steps',# evaluation strategy to adopt during training
                                        # `no`: No evaluation during training.
                                        # `steps`: Evaluate every `eval_steps`.
                                        # `epoch`: Evaluate every end of epoch.
            eval_steps = training_args['eval_steps'] or 300,           # evaluation step.
            dataloader_num_workers=4,
            load_best_model_at_end=True, # save_strategy, save_steps will be ignored
            metric_for_best_model="exact_match", # eval_accuracy
            greater_is_better=True, # set True if metric isn't loss
            label_smoothing_factor=0.5,
            fp16=True,
            do_train=True,
            do_eval=True,
            seed=42,
        )
        i= 0
        while os.path.exists(training_args.output_dir):
            training_args.output_dir= f'./result/{temp}{model_args.suffix}_{i}/'
            training_args.logging_dir= f'./logs/{temp}{model_args.suffix}_{i}/'
            i+=1

        joblist.append({"data_args": data_args, "model_args": model_args, "inf_args": inf_args, "training_args": training_args})

    return reservation_json['config'], joblist

get_reservation()
def build_reservation(path='./reservation/'):
    pass
