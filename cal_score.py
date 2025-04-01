import math

import torch
from networkx.algorithms.tournament import score_sequence

import IO
import glob
import numpy as np

eval_data_path = 'data/nq/nq-dev.json'
eval_data = IO.read_jsonl_to_list(eval_data_path)

# 파일 경로 지정
file_list = sorted(glob.glob("data/pkl/indexs*.pt"))  # 경로에 맞게 파일 찾기

# 텐서를 수직으로 쌓아서 (n*100, 10) 형태로 변환
tensor_list = []
for file in file_list:
    tensor = torch.load(file)
    tensor_list.append(tensor)

merged_tensor = torch.cat(tensor_list, dim=0)  # (200000, 765) 생성
torch.save(merged_tensor, 'data/pkl/merged_tensor.pt')
print(merged_tensor.shape)
min_list = []
score_list = []
idx = -1
for item in eval_data:
    idx += 1
    #print(item)
    question = item['question']
    docs = merged_tensor[idx].tolist()
    sum_score = 0.0
    _min = math.inf
    #print('docs:', docs)
    for doc in docs:
        #print(f'{doc}, score:', end=' ')
        score = 0.0
        for positive in item['positive_ctxs']:
            #print(type(positive), positive, end=' ')
            for index in positive['passage_id']:
                _min = min(_min, abs(int(doc) - int(index)))
                if (abs(int(doc)) - int(index)) <= 100:
                    score += positive['score']

        for negative in item['hard_negative_ctxs']:
            if str(doc) in negative['passage_id']:
                score -= negative['score']

        #print(score)
        sum_score += score
    #print(question,': Total score:', sum_score)
    abs(int(doc) - int(index))
    score_list.append(sum_score)
print(score_list)
# print(min_list)
np.save('data/pkl/scores.npy', np.array(score_list))
#
#
# import matplotlib.pyplot as plt
#
# # 샘플 데이터 (원하는 리스트로 변경 가능)
# data = min_list
#
# # x축: 인덱스, y축: 리스트 값
# plt.scatter(range(len(data)), data, color='b', marker='o', s=0.1, label="Data Points")
#
# # 그래프 설정
# plt.xlabel("Index")
# plt.ylabel("min")
# plt.title("1D List Plot")
# plt.legend()
# plt.grid(True)
#
# # 그래프 출력
# plt.show()

