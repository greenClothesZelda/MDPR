# Feature Map 관련 최적화 코드 추가
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
feature_map_path = '/feature_map.pt'  # ✅ Feature Map 저장 경로 추가


class Reference:
    def __init__(self, question_encoder, question_tokenizer, context_encoder, context_tokenizer, documents_PATH, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.question_encoder = question_encoder.to(self.device)
        self.question_tokenizer = question_tokenizer
        self.context_encoder = context_encoder.to(self.device)
        self.context_tokenizer = context_tokenizer
        self.batch_size = batch_size

        self.Q_past = torch.empty((0, 768), device=self.device)
        self.QA_list = torch.empty((0, 3), dtype=torch.long, device=self.device)

        if os.path.exists(tensor_path + Q_past_path):
            self.load_Q_past()
        else:
            self.save_Q_past()

        self.documents_PATH = documents_PATH
        self.docs = IO.load_passages(self.documents_PATH)
        self.passage_texts = [passage["text"] for passage in self.docs]

        # ✅ BM25를 생성하여 클래스 내부에서 저장
        self.bm25 = BM25Okapi(self.passage_texts)  # 🔹 여기서 미리 초기화

        if os.path.exists(tensor_path + feature_map_path):
            self.feature_map = torch.load(tensor_path + feature_map_path)
        else:
            self.feature_map = self.encode_passages_batch(self.passage_texts)
            torch.save(self.feature_map, tensor_path + feature_map_path)

        self.passage_embeddings = self.feature_map


        # ✅ Feature Map이 존재하면 로드, 없으면 생성
        if os.path.exists(tensor_path + feature_map_path):
            self.feature_map = torch.load(tensor_path + feature_map_path)
        else:
            self.feature_map = self.encode_passages_batch(self.passage_texts)
            torch.save(self.feature_map, tensor_path + feature_map_path)

        self.passage_embeddings = self.feature_map  # Feature Map 활용

    def encode_passages_batch(self, passages):
        """ ✅ 배치 단위로 DPR 문서 임베딩 생성 (Feature Map 적용) """
        all_embeddings = []
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for i in range(0, len(passages), self.batch_size):
            batch = passages[i:i + self.batch_size]  # 배치 단위로 슬라이싱
            passage_inputs = self.context_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)

            with torch.no_grad():
                embeddings = self.context_encoder(**passage_inputs).pooler_output

            all_embeddings.append(embeddings.cpu())  # GPU에서 CPU로 이동하여 저장

        return torch.cat(all_embeddings, dim=0)  # 모든 배치 결과를 합침

    def get_main_reference(self, embedded_query, k):
        """ ✅ DPR 우선 정렬 후 BM25 가중치 추가 """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sim_scores = []

        for i in range(0, len(self.passage_embeddings), self.batch_size):
            batch_embeddings = self.passage_embeddings[i:i + self.batch_size].to(device)
            sim = torch.matmul(batch_embeddings, embedded_query.T).squeeze()
            sim_scores.append(sim.cpu())

        combined_scores = torch.cat(sim_scores)  # 모든 배치 결과 합침

        # ✅ DPR로 상위 k개 추출
        top_k_values, top_k_indices = torch.topk(combined_scores, k, dim=0)

        # ✅ DPR로 선택된 문서에 대해서만 BM25 계산
        bm25_scores = torch.tensor(self.bm25.get_scores(embedded_query.squeeze().tolist()), dtype=torch.float32)

        # ✅ 가중치 조절 (BM25 영향 줄이기)
        weight_dpr, weight_bm25 = self.get_dynamic_weights(combined_scores)

        # ✅ DPR 문서에 대해 BM25 보조 점수 추가
        final_scores = weight_dpr * combined_scores[top_k_indices] + weight_bm25 * bm25_scores[top_k_indices]

        # ✅ 최종적으로 DPR 기반 정렬 유지 (BM25 영향 최소화)
        sorted_indices = torch.argsort(final_scores, descending=True)

        return top_k_indices[sorted_indices].tolist(), final_scores.tolist()


    def get_sub_references(self, embedded_query, a):
        if self.Q_past.shape[0] == 0:
            self.Q_past = embedded_query.view(1, -1)
            return []

        # 🔹 Q_past와 현재 쿼리 간 유사도 계산
        diff = self.Q_past - embedded_query.view(1, -1)
        similarity_scores = -torch.norm(diff, dim=1)  # 유사도가 높은 순 정렬

        # 🔹 top-a 인덱스 추출
        _, top_a_indices = torch.topk(similarity_scores, min(a, similarity_scores.shape[0]), dim=0)

        # 🔹 유효한 인덱스 필터링
        valid_indices = [idx for idx in top_a_indices.tolist() if idx < self.QA_list.shape[0]]

        return self.QA_list[valid_indices].squeeze().tolist() if valid_indices else []

    def get_reference(self, query, k):
        query_input = self.question_tokenizer(query, return_tensors="pt")
        query_input = {key: value.to(self.device) for key, value in query_input.items()}
        embedded_query = self.question_encoder(**query_input).pooler_output

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedded_query = embedded_query.to(device)

        # 🔹 get_main_reference에서 유사도와 인덱스를 함께 반환
        main_ref, main_scores = self.get_main_reference(embedded_query, k)

        # 🔹 `a` 값을 `k`의 절반으로 설정
        a = max(1, k // 2)  # `k`의 절반을 `a`로 선택 (최소 1)

        # 🔹 sub_ref에서 a개 가져오기
        sub_ref = self.get_sub_references(embedded_query, a)
        print(type(sub_ref))

        # 🔹 중복 제거하며 추가
        print(a)
        final_passages = {tuple(ref) if isinstance(ref, list) else ref for ref in sub_ref[:a]} # sub_ref에서 a개 추가
        remaining = k - len(final_passages)  # 부족한 개수 계산

        # k-a개 만큼 final_passages와 set(main_ref[:k-a]) 합집합
        final_passages = final_passages | set(main_ref[:remaining])

        # 🔹 부족한 경우 main_ref에서 추가
        for idx in main_ref:
            if len(final_passages) >= k:
                break
            final_passages.add(idx)  # 남은 개수만큼 main_ref에서 추가
        # print(f"\nQuery: {query}")
        # print(f"Main Ref: {main_ref}")
        # print(f"Final Passages (text):")
        # for idx in final_passages:
        #     print(f"  - {idx}: {self.passage_texts[idx][:100]}")


        return tuple(final_passages)  # ✅ 최종적으로 k개 반환



    def save_Q_past(self):
        torch.save(self.Q_past, tensor_path + Q_past_path)
        torch.save(self.QA_list, tensor_path + QA_list_path)

    def load_Q_past(self):
        self.Q_past = torch.load(tensor_path + Q_past_path, map_location=self.device)
        self.QA_list = torch.load(tensor_path + QA_list_path, map_location=self.device)

    def get_dynamic_weights(self, dpr_scores):
        """ ✅ 동적 가중치 적용: DPR 유사도가 높으면 DPR 비중을 높이고, 낮으면 BM25 비중 증가 """
        max_similarity = dpr_scores.max().item()

        if max_similarity >= 0.7:
            weight_dpr = 0.7
            weight_bm25 = 0.3
        elif max_similarity >= 0.5:
            weight_dpr = 0.5
            weight_bm25 = 0.5
        else:
            weight_dpr = 0.3
            weight_bm25 = 0.7

        return weight_dpr, weight_bm25

if __name__ == "__main__":
    ref = Reference(
        context_encoder=DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        context_tokenizer=DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        question_encoder=DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
        question_tokenizer=DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
        documents_PATH='downloads/data/wikipedia_split/psgs_w100.tsv',
        batch_size=128  # ✅ 배치 크기 설정
    )

    queries = ["Hello? my name is Jinsu", "Hello? my name is Alice"]

    for query in queries:
        references = ref.get_reference(query, 3)
        print(f"\n🔹 Query: {query}")
        print(f"🔹 References: {references}")
