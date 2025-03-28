# Feature Map 및 BM25 최적화 적용 버전 (BM25 비활성화됨)
import os
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
# from rank_bm25 import BM25Okapi  # 🔒 BM25 사용 안함
import option
from feature_map_manager import FeatureMapManager

# 텐서 저장 경로 및 파일명
tensor_path = option.tensor_path
Q_past_path = option.Q_past_path
QA_list_path = option.QA_list_path

class Reference:
    def __init__(self, question_encoder, question_tokenizer, context_encoder, context_tokenizer, documents_PATH, batch_size=10000):
        # 모델 및 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.question_encoder = question_encoder.to(self.device)
        self.question_tokenizer = question_tokenizer
        self.context_encoder = context_encoder.to(self.device)
        self.context_tokenizer = context_tokenizer
        self.batch_size = batch_size

        # 이전 질문 및 응답 저장 공간 초기화
        self.Q_past = torch.empty((0, 768), device=self.device)
        self.QA_list = torch.empty((0, 3), dtype=torch.long, device=self.device)

        # 이전 데이터 로드 또는 저장
        if os.path.exists(tensor_path + Q_past_path):
            self.load_Q_past()
        else:
            self.save_Q_past()

        # FeatureMapManager 클래스 초기화 및 feature_map 생성
        self.feature_manager = FeatureMapManager(
            context_encoder=self.context_encoder,
            context_tokenizer=self.context_tokenizer,
            documents_PATH=documents_PATH,
            batch_size=batch_size,
            tensor_path=tensor_path
        )

        # feature_map이 존재하지 않으면 새로 생성
        if not os.listdir(self.feature_manager.feature_map_dir):
            self.feature_manager.create_feature_maps()

        # BM25 사용 안 함
        # if not os.listdir(self.feature_manager.bm25_dir):
        #     self.feature_manager.create_bm25_passages()

        # DPR 문서 임베딩 전체 불러오기 (옵션)
        # self.passage_embeddings = self.feature_manager.load_all_feature_maps()

    def get_main_reference(self, embedded_query, k):
        """ 배치 단위로 저장된 feature_map 파일을 하나씩 불러와 DPR 유사도 계산 """
        sim_scores = torch.empty((embedded_query.shape[0], 21015324), dtype=torch.float, device=self.device)
    # feature_map 파일 순회하며 DPR 유사도 계산
        for i in range(self.feature_manager.get_num_feature_map_files()):
            #print(f"Calculating similarity for feature_map...{i}")
            feature_map = self.feature_manager.load_feature_map_by_index(i).to(self.device)
            with torch.no_grad():
                sim = torch.matmul(embedded_query, feature_map.T)
                sim_scores[:, 2000000*i:2000000*i+feature_map.shape[0]] = sim
            feature_map.cpu()  # 메모리 확보를 위해 CPU로 이동 및
            del feature_map
            torch.cuda.empty_cache()
        print(sim_scores.shape)

        top_k_values, top_k_indices = torch.topk(sim_scores, k, dim=1)

        # 유사도 기준 내림차순 정렬
        #sorted_indices = torch.argsort(top_k_values, descending=True)
        #print(top_k_indices[sorted_indices].tolist(), top_k_values.tolist())
        return top_k_indices, top_k_values

    def get_sub_references(self, embedded_queries, a):
        """
        🔹 (n, 768) 형태의 쿼리 벡터를 입력받아 각 쿼리에 대해
           Q_past와의 유사도를 기반으로 보조 참조 인덱스를 반환합니다.
        🔹 결과는 (n, a) 형태의 리스트로 반환됩니다.
        """
        if self.Q_past.shape[0] == 0:
            self.Q_past = embedded_queries.clone()
            return [[] for _ in range(embedded_queries.shape[0])]  # 쿼리 개수만큼 빈 리스트

        sub_refs_all = []

        for embedded_query in embedded_queries:
            # 🔹 유사도 계산
            diff = self.Q_past - embedded_query.view(1, -1)
            similarity_scores = -torch.norm(diff, dim=1)

            # 🔹 top-a 유사한 인덱스
            _, top_a_indices = torch.topk(similarity_scores, min(a, similarity_scores.shape[0]), dim=0)

            # 🔹 QA_list 범위 내에서만 인덱스 유효성 확인
            valid_indices = [idx for idx in top_a_indices.tolist() if idx < self.QA_list.shape[0]]

            # 🔹 보조 참조 인덱스 추출 (-1은 제외)
            sub_refs = [self.QA_list[idx].tolist()[0] for idx in valid_indices if self.QA_list[idx][0] >= 0]
            sub_refs_all.append(sub_refs)

        return sub_refs_all  # List[List[int]] (n, a)


    def get_reference(self, embedded_query, k):
        """
        🔹 (n, 768) 형태의 배치 쿼리 DPR 임베딩을 받아,
           각 쿼리에 대해 DPR 기반으로 top-k 문서 인덱스를 반환합니다.
        🔹 현재 보조 참조(sub_ref)는 비활성화 상태 (a=0)
        🔹 반환값: List[List[int]] (쿼리 개수 n, 각 쿼리당 k개 인덱스)
        """
        n = embedded_query.shape[0]

        # 🔹 DPR 유사도 기반 메인 참조
        main_ref_passages, _ = self.get_main_reference(embedded_query, k)  # (n, k)

        # 🔹 보조 참조는 현재 사용 안 함 (a=0)
        a = 2
        sub_ref_passages = [[] for _ in range(n)]  # future-proof structure

        final_refs_passages = []

        for i in range(n):
            main_ref = main_ref_passages[i]
            sub_ref = sub_ref_passages[i]  # 현재는 빈 리스트이지만 확장 가능

            # 🔸 -1 제거 및 중복 방지
            final_passages = {
                ref if isinstance(ref, int) else ref[0]
                for ref in sub_ref[:a]
                if (ref if isinstance(ref, int) else ref[0]) >= 0
            }

            # 🔸 부족한 개수만큼 main_ref에서 보충
            remaining = k - len(final_passages)
            final_passages.update(main_ref[:remaining])

            for idx in main_ref:
                if len(final_passages) >= k:
                    break
                final_passages.add(idx)

            final_refs_passages.append(list(final_passages))

            # 🔹 Q_past 및 QA_list 업데이트 (main_ref의 첫 번째만 기록)
            self.Q_past = torch.cat([self.Q_past, embedded_query[i].unsqueeze(0).detach()], dim=0)
            new_QA = torch.tensor([[main_ref[0], -1, -1]], dtype=torch.long).to(self.device)
            self.QA_list = torch.cat([self.QA_list, new_QA], dim=0)

        self.save_Q_past()
        return final_refs_passages  # ✅ List[List[int]] (n, k)



    def save_Q_past(self):
        """ 이전 쿼리 임베딩 및 관련 문서 저장 """
        torch.save(self.Q_past, tensor_path + Q_past_path)
        torch.save(self.QA_list, tensor_path + QA_list_path)

    def load_Q_past(self):
        """ 이전 쿼리 임베딩 및 관련 문서 로드 """
        self.Q_past = torch.load(tensor_path + Q_past_path, map_location=self.device)
        self.QA_list = torch.load(tensor_path + QA_list_path, map_location=self.device)

    def get_dynamic_weights(self, dpr_scores):
        """ BM25 가중치 적용용 함수 (현재 미사용) """
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
        documents_PATH='data/docs/psgs_w100.tsv'
    )

    queries = ["Hello? my name is Jinsu", "Hello? my name is Alice"]
    for query in queries:
        refs = ref.get_reference(query, 3)
        print(f"\n🔹 Query: {query}\n🔹 References: {refs}")
