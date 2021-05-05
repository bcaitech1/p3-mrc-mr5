import random
import numpy as np
from pprint import pprint

from datasets import load_dataset, load_metric, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer
)
import os
print(os.path.exists("./"))
print(os.listdir("./"))
# dataset = load_dataset("squad_kor_v1")
dataset = load_from_disk('./data/train_dataset')

corpus = list(set([example['context'] for example in dataset['train']]))
corpus.extend(list(set([example['context'] for example in dataset['validation']])))
tokenizer_func = lambda x : x.split(' ')

vectorizer = TfidfVectorizer(tokenizer=tokenizer_func, ngram_range=(1,2))
sp_matrix = vectorizer.fit_transform(corpus)

def get_relevant_doc(vectorizer, query, k=1):
    """
    vocab에 없는 이상한 단어로 query시 assetion 발생
    """
    query_vec = vectorizer.transform([query])
    assert np.sum(query_vec) != 0, "vocab에 없는 이상한 단어"
    result = query_vec * sp_matrix.T
    sorted_result = np.argsort(-result.data)
    doc_scores = result.data[sorted_result]
    doc_ids = result.indices[sorted_result]
    return doc_scores[:k], doc_ids[:k]

query = "미국의 대통령은 누구인가?"
_, doc_id = get_relevant_doc(vectorizer, query, k=1)

# print("******Result********")
# print("[Searc query]\n", query, "\n")
# print(f"[Relevant Doc ID(Top 1 passage)]: {doc_id.item()}")
# print(corpus[doc_id.item()])

model_name= "./result/bert-base-multilingual-case/checkpoint-900/"
config = AutoConfig.from_pretrained(
    model_name
)
mrc_model = AutoModelForQuestionAnswering.from_pretrained(
    model_name,
    from_tf=bool(".ckpt" in model_name),
    config=config,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True
)
mrc_model = mrc_model.eval()

def get_answer_form_context(context, question, model, tokenizer):
    encoded_dict = tokenizer.encode_dict(
        question,
        context,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    non_padded_ids = encode_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
    full_text =tokenizer.decode(non_padded_ids)
    inputs = {
        'input_ids' : torch.tensor([encoded_dict['input_ids']], dtype=torch.long),
        'attention_mask' : torch.tensor([encoded_dict['attention_mask']], dtype=torch.long),
        'token_type_ids' : torch.tensor([encoded_dict['token_type_ids']], dtype=torch.long)
    }

    outputs = model(**inputs)
    start, end = torch.amx(outputs.start_logits, axis=1).indices.item(), torch.max(outputs.end_logits, axis=1).indices.item()
    answer = tokenizer.decode(encoded_dict['input_ids'][start:end+1])
    return answer

context = corpus[doc_id.item()]

answer = get_answer_form_context(context, question, mrc_model, tokenizer)
print(answer)
