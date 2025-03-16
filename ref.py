# 컨텍스트 인코더 및 토크나이저 로드
from rank_bm25 import BM25Okapi
from transformers import DPRContextEncoderTokenizer, DPRContextEncoder, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import IO
import torch
import os

import option

# 텐서 저장 경로
tensor_path = 'data/tensor'
Q_past_path = '/Q_past.pt'
QA_list_path = '/QA_list.pt'

class Reference:
    def __init__(self, question_encoder, question_tokenizer, context_encoder, context_tokenizer, documents_PATH):
        self.question_encoder = question_encoder
        self.question_tokenizer = question_tokenizer
        self.context_encoder = context_encoder
        self.context_tokenizer = context_tokenizer

        self.Q_past = torch.empty((0, 768))  # 임베딩 차원 맞춰 초기화
        self.QA_list = torch.empty((0, 3), dtype=torch.long)  # 저장된 passage 인덱스 초기화

        # 파일이 존재하는 경우 텐서 로드
        if os.path.exists(tensor_path + Q_past_path):
            self.load_Q_past()
        else:
            self.save_Q_past()

        self.documents_PATH = documents_PATH
        self.docs = IO.load_passages(self.documents_PATH)
        self.passage_texts = [passage["text"] for passage in self.docs]

        # DPR 컨텍스트 인코더를 사용하여 패시지 임베딩 생성
        passage_inputs = self.context_tokenizer(self.passage_texts, padding=True, truncation=True, return_tensors="pt")
        self.passage_embeddings = self.context_encoder(**passage_inputs).pooler_output

    def get_main_reference(self, embedded_query, k):
        # DPR 유사도 계산
        sim = torch.matmul(self.passage_embeddings, embedded_query.T).squeeze()

        # BM25 점수 계산
        bm25 = BM25Okapi(self.passage_texts)
        bm25_scores = torch.tensor(bm25.get_scores(embedded_query.squeeze().tolist()), dtype=torch.float32)

        # DPR 유사도 + BM25 점수 결합
        combined_scores = sim + option.weight_bm25 * bm25_scores

        # 결합된 점수를 기반으로 상위 k 인덱스 가져오기
        top_k_combined_values, top_k_combined_indices = torch.topk(combined_scores, k, dim=0)

        return top_k_combined_indices.tolist()

    def get_sub_references(self, embedded_query, k):
        if self.Q_past.shape[0] == 0:
            self.Q_past = embedded_query.view(1, -1)
            return []

        # 현재 쿼리(embedded_query)와 Q_past(이전 질문들) 간 유사도 계산
        diff = self.Q_past - embedded_query.view(1, -1)
        similarity_scores = -torch.norm(diff, dim=1)  # 유사도가 높은 순으로 정렬

        # 상위 k개의 유사도가 높은 질문 인덱스 가져오기
        top_k_values, top_k_indices = torch.topk(similarity_scores, min(k, similarity_scores.shape[0]), dim=0)

        # ✅ 인덱스가 QA_list 범위를 초과하지 않도록 필터링
        valid_indices = [idx for idx in top_k_indices.tolist() if idx < self.QA_list.shape[0]]

        # ✅ self.QA_list가 비어 있으면 빈 리스트 반환하여 오류 방지
        if len(valid_indices) > 0:
            sub_references = self.QA_list[valid_indices].squeeze().tolist()  # ✅ 리스트 내부 리스트 제거
        else:
            sub_references = []

        # Q_past 업데이트
        self.Q_past = torch.cat((self.Q_past, embedded_query.view(1, -1)), dim=0)

        return sub_references



    def get_reference(self, query, k):
        query_input = self.question_tokenizer(query, return_tensors="pt")
        embedded_query = self.question_encoder(**query_input).pooler_output

        main_ref = self.get_main_reference(embedded_query, k)
        sub_ref = self.get_sub_references(embedded_query, k)

        # ✅ 중복 제거 (set을 사용하여 리스트 내부 원소 비교)
        unique_passages = set(p[0] if isinstance(p, list) else p for p in main_ref)
        final_passages = []
        pointer = 0

        while pointer < len(sub_ref):
            passage = sub_ref[pointer]
            passage_id = passage[0] if isinstance(passage, list) else passage  # 리스트 내부 원소 비교
            if passage_id not in unique_passages:
                unique_passages.add(passage_id)
                final_passages.append(passage)
            pointer += 1

        # ✅ 부족한 passage 개수 계산
        n = k - len(final_passages)

        # ✅ 부족한 passage를 유사도가 높은 passage로 채우기
        if n > 0:
            remaining_indices = list(set(range(len(self.passage_embeddings))) - unique_passages)
            if len(remaining_indices) > 0:
                remaining_scores = torch.matmul(self.passage_embeddings[remaining_indices], embedded_query.T).squeeze()
                sorted_remaining_indices = torch.argsort(remaining_scores, descending=True)[:n].tolist()

                for idx in sorted_remaining_indices:
                    final_passages.append(remaining_indices[idx])

        return main_ref, final_passages  # ✅ 리스트 내부 리스트 제거 후 반환




    def query_embedding(self, query):
        query_input = self.question_tokenizer(query, return_tensors="pt")
        query_embedding = self.question_encoder(**query_input).pooler_output
        return query_embedding

    def save_Q_past(self):
        torch.save(self.Q_past, tensor_path + Q_past_path)
        torch.save(self.QA_list, tensor_path + QA_list_path)

    def load_Q_past(self):
        self.Q_past = torch.load(tensor_path + Q_past_path)
        self.QA_list = torch.load(tensor_path + QA_list_path)

if __name__ == "__main__":
    ref = Reference(
        context_encoder=DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        context_tokenizer=DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        question_encoder=DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
        question_tokenizer=DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
        documents_PATH='data/docs/psgs_w100.tsv'
    )

    # ✅ 최종 실행 코드 (중복 제거된 passage들이 정확히 k개 반환되는지 확인)
    queries = [
        "Hello? my name is Jinsu",
        "Hello? my name is John",
        "Hello? my name is Alice",
        "Hello? my name is Bob",
        "Hello? my name is Charlie"
    ]

    for query in queries:
        references = ref.get_reference(query, 3)
        print(f"\n🔹 Query: {query}")
        print(f"🔹 References: {references}")
