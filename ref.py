# Feature Map 및 BM25 최적화 적용 버전
import os
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from rank_bm25 import BM25Okapi
import option
from feature_map_manager import FeatureMapManager

# 텐서 저장 경로 및 파일명
tensor_path = 'data/tensor'
Q_past_path = '/Q_past.pt'
QA_list_path = '/QA_list.pt'

class Reference:
    def __init__(self, question_encoder, question_tokenizer, context_encoder, context_tokenizer, documents_PATH, batch_size=32):
        # 디바이스 설정 (GPU 우선 사용)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 인코더 및 토크나이저 설정
        self.question_encoder = question_encoder.to(self.device)
        self.question_tokenizer = question_tokenizer
        self.context_encoder = context_encoder.to(self.device)
        self.context_tokenizer = context_tokenizer
        self.batch_size = batch_size

        # 과거 쿼리 및 레퍼런스 초기화
        self.Q_past = torch.empty((0, 768), device=self.device)
        self.QA_list = torch.empty((0, 3), dtype=torch.long, device=self.device)

        # 과거 쿼리 정보 불러오기 (또는 저장)
        if os.path.exists(tensor_path + Q_past_path):
            self.load_Q_past()
        else:
            self.save_Q_past()

        # FeatureMapManager 초기화 및 처리
        self.feature_manager = FeatureMapManager(
            context_encoder=self.context_encoder,
            context_tokenizer=self.context_tokenizer,
            documents_PATH=documents_PATH,
            batch_size=batch_size,
            tensor_path=tensor_path
        )

        # DPR 임베딩 및 BM25 전처리가 되어 있지 않으면 생성
        if not os.listdir(self.feature_manager.feature_map_dir):
            self.feature_manager.create_feature_maps()
        if not os.listdir(self.feature_manager.bm25_dir):
            self.feature_manager.create_bm25_passages()

        # DPR 임베딩과 BM25 객체 불러오기
        self.passage_embeddings = self.feature_manager.load_all_feature_maps()
        self.bm25 = self.feature_manager.load_bm25()

    def encode_passages_batch(self, passages):
        # 배치 단위로 DPR 임베딩 생성
        all_embeddings = []
        for i in range(0, len(passages), self.batch_size):
            batch = passages[i:i + self.batch_size]
            passage_inputs = self.context_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.context_encoder(**passage_inputs).pooler_output
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def get_main_reference(self, embedded_query, k):
        # DPR 임베딩 기반 유사도 계산 후 BM25 점수와 가중 평균하여 상위 k개 추출
        sim_scores = []
        for i in range(0, len(self.passage_embeddings), self.batch_size):
            batch_embeddings = self.passage_embeddings[i:i + self.batch_size].to(self.device)
            sim = torch.matmul(batch_embeddings, embedded_query.T).squeeze()
            sim_scores.append(sim.cpu())
        combined_scores = torch.cat(sim_scores)

        # DPR top-k 인덱스 추출
        top_k_values, top_k_indices = torch.topk(combined_scores, k, dim=0)

        # BM25 점수 계산
        bm25_scores = torch.tensor(self.bm25.get_scores(embedded_query.squeeze().tolist()), dtype=torch.float32)

        # 가중치 계산 및 최종 점수
        weight_dpr, weight_bm25 = self.get_dynamic_weights(combined_scores)
        final_scores = weight_dpr * combined_scores[top_k_indices] + weight_bm25 * bm25_scores[top_k_indices]

        # 최종 점수 기준 정렬
        sorted_indices = torch.argsort(final_scores, descending=True)
        return top_k_indices[sorted_indices].tolist(), final_scores.tolist()

    def get_sub_references(self, embedded_query, a):
        # 과거 쿼리 기반 유사도 측정하여 관련 문서 인덱스 반환
        if self.Q_past.shape[0] == 0:
            self.Q_past = embedded_query.view(1, -1)
            return []
        diff = self.Q_past - embedded_query.view(1, -1)
        similarity_scores = -torch.norm(diff, dim=1)
        _, top_a_indices = torch.topk(similarity_scores, min(a, similarity_scores.shape[0]), dim=0)
        valid_indices = [idx for idx in top_a_indices.tolist() if idx < self.QA_list.shape[0]]
        return self.QA_list[valid_indices].squeeze().tolist() if valid_indices else []

    def get_reference(self, query, k):
        # 쿼리를 DPR 임베딩하고, 관련 문서 k개 반환
        query_input = self.question_tokenizer(query, return_tensors="pt")
        query_input = {key: value.to(self.device) for key, value in query_input.items()}
        embedded_query = self.question_encoder(**query_input).pooler_output.to(self.device)

        # 메인 및 서브 레퍼런스 추출
        main_ref, _ = self.get_main_reference(embedded_query, k)
        a = max(1, k // 2)
        sub_ref = self.get_sub_references(embedded_query, a)

        # 중복 제거 및 부족분 보완
        final_passages = {ref if isinstance(ref, int) else ref[0] for ref in sub_ref[:a]}
        remaining = k - len(final_passages)
        final_passages = final_passages | set(main_ref[:remaining])
        for idx in main_ref:
            if len(final_passages) >= k:
                break
            final_passages.add(idx)

        # 과거 쿼리 및 레퍼런스 저장
        self.Q_past = torch.cat([self.Q_past, embedded_query.detach()], dim=0)
        new_QA = torch.tensor([[idx, -1, -1] for idx in main_ref[:1]], dtype=torch.long).to(self.device)
        self.QA_list = torch.cat([self.QA_list, new_QA], dim=0)
        self.save_Q_past()

        return list(final_passages)

    def save_Q_past(self):
        # 과거 쿼리 및 레퍼런스 저장
        torch.save(self.Q_past, tensor_path + Q_past_path)
        torch.save(self.QA_list, tensor_path + QA_list_path)

    def load_Q_past(self):
        # 과거 쿼리 및 레퍼런스 불러오기
        self.Q_past = torch.load(tensor_path + Q_past_path, map_location=self.device)
        self.QA_list = torch.load(tensor_path + QA_list_path, map_location=self.device)

    def get_dynamic_weights(self, dpr_scores):
        # DPR 유사도에 따라 가중치 조절
        max_similarity = dpr_scores.max().item()
        if max_similarity >= 0.7:
            return 0.7, 0.3
        elif max_similarity >= 0.5:
            return 0.5, 0.5
        else:
            return 0.3, 0.7


if __name__ == "__main__":
    ref = Reference(
        context_encoder=DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        context_tokenizer=DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        question_encoder=DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
        question_tokenizer=DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
        documents_path='data/docs/psgs_w100.tsv'
    )

    queries = ["Hello? my name is Jinsu", "Hello? my name is Alice"]
    for query in queries:
        refs = ref.get_reference(query, 3)
        print(f"\n🔹 Query: {query}\n🔹 References: {refs}")
