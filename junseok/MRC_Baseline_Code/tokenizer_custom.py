# from kobert_tokenizer import KoBertTokenizer

# from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
# import os
# tokenizer = AutoTokenizer.from_pretrained(
#     "facebook/mbart-large-cc25",
#     use_fast=True,
# )
# tokenizer = KoBertTokenizer.from_pretrained(
#     "monologg/kobert",
#     use_fast=True,
# )

# print(dir(tokenizer))
# # tokenizer
# encoded = tokenizer.encode('3살 즈음에 도쿠가와 요시무네의 증손 도요치요(豊千代, 훗날의 이에나리)와 약혼하고 시게히메로 이름을 바꾸었다. 약혼 직후 에도의 히토쓰바시 저택으로 이주하여 약혼자와 함께 자랐지만, 10대 쇼군 아들의 급서로 토요치요가 후계자가 되자 이 약혼이 문제가 되었다. 원래 쇼군의 정실은 3대 이에미츠 이후 셋케나 미야케(친왕가)에서 맞이하는 관례가 있어 도자마 다이묘의 딸이 정실이 된다는 것은 전례에 없던 일이었다. 이 때, 5대 사쓰마 번주의 정실이며 도쿠가와 쓰나요시의 양녀인 조간인(다케히메)이 약혼을 추진했다는 시게히데의 주장으로 인해 막부측에서도 약혼 13년 만에 혼인을 받아들이게 되었다. 하지만, 쇼군 정실에 대한 관례를 존중하여 시마즈가의 친척인 코노에 츠네히로의 양녀가 되어 「고노에 다다코(近衛寔子)」라는 이름으로 결혼하였다. 시마즈 시게히데의 정실 다모쓰히메가 이에나리의 고모인 관계로 남편과 시게히메는 사촌형제뻘이 되기도 했다.\n\n이 결혼으로 인해 시마즈 시게히데를 비릇하여 오토세노카타의 친족인 이치다 씨족까지 권세를 쥐게 되었다. 오토세노카타 역시 시마즈 나리노부의 생모를 포함한 다른 측실들을 에도저택에서 내쫓고 시게히데의 정실 행세를 하는 등의 횡포를 부려 훗날 근사록 붕괴의 중요 원인을 제공하였다.\n\n1796년 쇼군의 5남 도쿠가와 아쓰노스케를 낳아 시게히메 부녀의 권세는 더욱 위풍당당하게 되었다. 2대 쇼군 이래로 처음 태어난 정실 소생의 아들이었지만 이미 측실 소생의 민지로(후의 이에요시)가 후계자로 정해진 관계로 아쓰노스케는 시미즈 도쿠가와 가(고산쿄)의 양자가 되었다. 그러나 아쓰노스케는 3년 후에 요절하고 1798년에도 임신했으나 유산하게 된다. 더욱이 교만한 행동으로 인해 이에나리의 총애마저 잃게 되어 더이상 아이를 낳지 못하게 된다.\n\n이복남동생이자 9대 번주인 시마즈 나리노부가 강제은거된 후 재정난을 이유로 여러 번 귀향을 신청했으나 막부의 허락을 받지 못했다. 그 이유는 시게히메의 생모 사후 시마즈 나리오키가 이치다 일족을 사츠마에서 몰아낸 것에 격노하여 미다이도코로의 권위를 이용해 영향력을 행사한 결과라고 한다. 1837년 은거한 남편과 함께 니시노마루로 옮기고 「오오미다이도코로(大御台所)」라고 불리게 된다. 1841년 남편의 사망으로 인해 삭발출가하여 「고다이인(広大院)」이라는 계명을 받았다. 다음 해에는 종1위의 관직 등급을 받게되어 이후 「이치이사마」이라고 불리게 된다. 만년에 와서 이에나리의 측실이었던 오미요노카타가 자기 소생의 딸이 낳은 마에다 요시야스(14대 가가 번주)를 쇼군으로 옹립하려는 음모를 저지하는 활약을 하기도 했다.\n\n1844년 사망. 묘소는 조조지(増上寺).')

# # print(tokenizer.decode(encoded))
# for token in encoded:
#     print(tokenizer.decode(token), end=", ")
# def get_recent_model():
#     models_dir = './result'
#     all_models = [models_dir+'/'+d for d in os.listdir(models_dir) if os.path.isdir(models_dir+'/'+d)]
#     latest_models = max(all_models, key=os.path.getmtime)
#     all_checkpoints = [latest_models+'/'+d for d in os.listdir(latest_models) if os.path.isdir(latest_models+'/'+d)]
#     latest_checkpoints = max(all_checkpoints, key=os.path.getmtime)
#     return latest_models.replace(models_dir+'/', ''), latest_checkpoints

# # print(get_recent_model())
# # print('dfs'.replace('d',''))
# from sklearn.feature_extraction.text import TfidfVectorizer
# corpus = ['This is the first document.',
#           'This document is the second document.',
#           'And this is the third one.',
#           'Is this the first document?',
#           ]
# vectorizer = TfidfVectorizer(max_features=2)
# X = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names())
# print(X[0,:])
# print("*************")
# print(X[1,:])
# print("*************")
# print(X[2,:])
# print("*************")
# print(X[3,:])

# """ Implementation of OKapi BM25 with sklearn's TfidfVectorizer
# Distributed as CC-0 (https://creativecommons.org/publicdomain/zero/1.0/)
# """

# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy import sparse


# class BM25(object):
#     def __init__(self, b=0.75, k1=1.6):
#         self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
#         self.b = b
#         self.k1 = k1

#     def fit(self, X):
#         """ Fit IDF to documents X """
#         self.vectorizer.fit(X)
#         y = super(TfidfVectorizer, self.vectorizer).transform(X)
#         self.avdl = y.sum(1).mean()

#     def transform(self, q, X):
#         """ Calculate BM25 between query q and documents X """
#         b, k1, avdl = self.b, self.k1, self.avdl

#         # apply CountVectorizer
#         X = super(TfidfVectorizer, self.vectorizer).transform(X)
#         len_X = X.sum(1).A1
#         q, = super(TfidfVectorizer, self.vectorizer).transform([q])
#         assert sparse.isspmatrix_csr(q)

#         # convert to csc for better column slicing
#         X = X.tocsc()[:, q.indices]
#         denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
#         # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
#         # to idf(t) = log [ n / df(t) ] with minus 1
#         idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
#         numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
#         return (numer / denom).sum(1).A1



# #------------ End of library impl. Followings are the example -----------------

# from sklearn.datasets import fetch_20newsgroups


# # texts = fetch_20newsgroups(subset='train').data
# # bm25 = BM25()
# # bm25.fit(texts[5:])
# # print(bm25.transform(texts[:5], texts))
# test = [i for i in range(103)]
# for i in range(0,len(test),10):
#     print(test[i:i+10])

import numpy as np
# a = [[1, 2, 3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]
# a = np.asarray(a)
# print(a.shape)
# print(a[:1,1])
# print(a[:1,None])

a = [1,2,3]
print(np.indices((2,3)))