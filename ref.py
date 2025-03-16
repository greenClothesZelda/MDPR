# 컨텍스트 인코더 및 토크나이저 로드 (strict=False 추가하여 경고 방지)
from rank_bm25 import BM25Okapi
from transformers import DPRContextEncoderTokenizer, DPRContextEncoder, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import IO
import torch
import os

import option

tensor_path= 'data/tensor'
Q_past_path = '/Q_past.pt'
QA_list_path = '/QA_list.pt'

class Reference:
    def __init__(self, question_encoder, question_tokenizer, context_encoder, context_tokenizer, documents_PATH):
        self.question_encoder = question_encoder
        self.question_tokenizer = question_tokenizer
        self.context_encoder = context_encoder
        self.context_tokenizer = context_tokenizer

        self.Q_past = torch.tensor([])
        self.QA_list = torch.tensor([])

        # 파일이 존재하는지 확인
        if os.path.exists(tensor_path+Q_past_path):
            # 파일이 존재하면 텐서를 로드
            self.load_Q_past()
        else:
            self.save_Q_past()



        self.documents_PATH = documents_PATH
        self.docs = IO.load_passages(self.documents_PATH)
        self.passage_texts = [passage["text"] for passage in self.docs]
        passage_inputs = self.context_tokenizer(self.passage_texts, padding=True, truncation=True, return_tensors="pt")
        self.passage_embeddings = self.context_encoder(**passage_inputs).pooler_output

    def get_main_reference(self, embedded_query, k):
        # 유사도 점수 계산
        sim = torch.matmul(self.passage_embeddings, embedded_query.T)

        # BM25 점수 계산
        bm25 = BM25Okapi(self.passage_texts)
        bm25_scores = bm25.get_scores(embedded_query.squeeze().tolist())

        # 유사도 점수와 BM25 점수 결합
        combined_scores = sim.squeeze().tolist() + option.weight_bm25 * bm25_scores

        # 결합된 점수를 기반으로 상위 k 인덱스 가져오기
        combined_scores_tensor = torch.tensor(combined_scores)
        top_k_combined_values, top_k_combined_indices = torch.topk(combined_scores_tensor, k, dim=0)

        # QA 리스트 업데이트
        if self.QA_list.numel() == 0:  # QA_list가 비어 있는 경우 초기화
            self.QA_list = top_k_combined_indices.unsqueeze(0)
        else:
            self.QA_list = torch.cat((self.QA_list, top_k_combined_indices.unsqueeze(0)), dim=0)

        #print(self.QA_list)

        return top_k_combined_indices.tolist()

    def get_reference(self, query, k):
        query_input = self.question_tokenizer(query, return_tensors="pt")
        embedded_query = self.question_encoder(**query_input).pooler_output
        main_ref = self.get_main_reference(embedded_query, k)
        sub_ref = self.get_sub_references(embedded_query, k, main_ref)
        return main_ref, sub_ref

    def get_sub_references(self, embedded_query, k, main_ref):
        if k > self.Q_past.shape[0]:
            self.Q_past = torch.cat((self.Q_past, embedded_query), dim=0)
            return []

        diff = self.Q_past - embedded_query
        self.Q_past = torch.cat((self.Q_past, embedded_query), dim=0)
        diff = torch.norm(diff, dim=1)
        top_k_values, top_k_indices = torch.topk(diff, k, dim=0)
        return self.QA_list[top_k_indices][0]


    def query_embedding(self, query):
        query_input = self.question_tokenizer(query, return_tensors="pt")
        query_embedding = self.question_encoder(**query_input).pooler_output
        return query_embedding

    def save_Q_past(self):
        torch.save(self.Q_past, tensor_path+Q_past_path)
        torch.save(self.QA_list, tensor_path+QA_list_path)

    def load_Q_past(self):
        self.Q_past = torch.load(tensor_path+Q_past_path)
        self.QA_list = torch.load(tensor_path+QA_list_path)

if __name__ == "__main__":
    ref = Reference(
        context_encoder=DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        context_tokenizer=DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        question_encoder=DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
        question_tokenizer=DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
        documents_PATH='data/docs/psgs_w100.tsv'
    )
    print(ref.passage_embeddings)
    print(ref.get_reference('Hello? my nfdasdfasafame is jinsu', 2))
    print(ref.get_reference('Hello? myasfdsa name is jinsu', 2))
    print(ref.get_reference('Hello? my dsafsdafname is jinsu', 2))
    print(ref.get_reference('Hello? my nsdfadsaame is jinsu', 2))
    print(ref.get_reference('Hello? mydfasfa name is jinsu', 2))