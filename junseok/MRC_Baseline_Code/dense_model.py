# google drive 에 올려둔 미리 학습해둔 인코더 불러오기
from datasets import load_dataset
from transformers import BertModel, BertPreTrainedModel, BertConfig, AutoTokenizer


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooled_output = outputs[1]

        return pooled_output


model_checkpoint = "bert-base-multilingual-cased"
p_encoder = BertEncoder.from_pretrained(model_checkpoint).cuda()
q_encoder = BertEncoder.from_pretrained(model_checkpoint).cuda()
model_dict = torch.load("./dense_encoder/encoder.pth")
p_encoder.load_state_dict(model_dict['p_encoder'])
q_encoder.load_state_dict(model_dict['q_encoder'])

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

dataset = load_dataset("squad_kor_v1")


def to_cuda(batch):
    return tuple(t.cuda() for t in batch)

def get_relevant_doc(p_encoder, q_encoder, query, k=1):
    with torch.no_grad():
        p_encoder.eval()
        q_encoder.eval()

        q_seqs_val = tokenizer(
            [query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
        q_emb = q_encoder(**q_seqs_val).to('cpu')  # (num_query, emb_dim)

        p_embs = []
        for p in valid_corpus:
            p = tokenizer(p, padding="max_length",
                            truncation=True, return_tensors='pt').to('cuda')
            p_emb = p_encoder(**p).to('cpu').numpy()
            p_embs.append(p_emb)

    p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)
    dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))

    rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

    return dot_prod_scores.squeeze(), rank[:k]

random.seed(2020)
valid_corpus = list(set([example['context'] for example in dataset['validation']]))[:10]
sample_idx = random.choice(range(len(dataset['validation'])))
query = dataset['validation'][sample_idx]['question']
ground_truth = dataset['validation'][sample_idx]['context']

if not ground_truth in valid_corpus:
  valid_corpus.append(ground_truth)

print("{} {} {}".format('*'*20, 'Ground Truth','*'*20))
print("[Search query]\n", query, "\n")
pprint(ground_truth, compact=True)

# valid_corpus

_, doc_id = get_relevant_doc(p_encoder, q_encoder, query, k=1)

""" 상위 1개 문서를 추출했을 때 결과 확인 """
print("{} {} {}".format('*'*20, 'Result','*'*20))
print("[Search query]\n", query, "\n")
print(f"[Relevant Doc ID(Top 1 passage)]: {doc_id}")
print(valid_corpus[doc_id.item()])
# print(answer)

""" 상위 5개를 추출하여 점수 확인 """
dot_prod_scores, rank = get_relevant_doc(p_encoder, q_encoder, query, k=5)

for i in range(5):
    print(rank[i])
    print("Top-%d passage with score %.4f" % (i+1, dot_prod_scores.squeeze()[rank[i]]))
    print(valid_corpus[rank[i]])