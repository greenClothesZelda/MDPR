import os
import torch
import pandas as pd
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
# from rank_bm25 import BM25Okapi  # 🔒 BM25 사용 비활성화
import IO

class FeatureMapManager:
    def __init__(self, context_encoder, context_tokenizer, documents_PATH, batch_size=500, tensor_path=r"C:\Users\wlstn\.cache\kagglehub\datasets\tensor"):
        # DPR Context 인코더 및 토크나이저 설정
        self.context_encoder = context_encoder
        self.context_tokenizer = context_tokenizer

        # 문서 데이터 경로 및 배치 설정
        self.documents_PATH = documents_PATH
        self.batch_size = batch_size
        self.tensor_path = tensor_path

        # Feature map 저장 디렉토리 설정
        self.feature_map_dir = os.path.join(tensor_path, 'feature_maps')
        os.makedirs(self.feature_map_dir, exist_ok=True)
        # self.bm25_dir = os.path.join(tensor_path, 'bm25_passages')  # 🔒 BM25 디렉토리 비활성화
        # os.makedirs(self.bm25_dir, exist_ok=True)  # 🔒 BM25 디렉토리 생성 비활성화

    def create_feature_maps(self):
        """DPR 인코더를 사용해 배치 단위로 문서를 임베딩하고 파일로 저장"""
        for i, chunk in enumerate(IO.load_passages_in_chunks(self.documents_PATH, self.batch_size)):
            print(f"Creating feature map {i+30933}...")
            texts = [p['text'] for p in chunk]
            inputs = self.context_tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: v.to(self.context_encoder.device) for k, v in inputs.items()}
            with torch.no_grad():
                embeddings = self.context_encoder(**inputs).pooler_output.cpu()
            torch.save(embeddings, os.path.join(self.feature_map_dir, f'feature_map_{i+30933}.pt'))
            # 🔽 메모리 확보
            torch.cuda.empty_cache()  # ⚠️ GPU에서 메모리 완전 해제 (옵션)

    # def create_bm25_passages(self):
    #     for i, chunk in enumerate(self.load_passages_in_chunks()):
    #         texts = chunk['text'].tolist()
    #         with open(os.path.join(self.bm25_dir, f'bm25_passages_{i}.txt'), 'w', encoding='utf-8') as f:
    #             for passage in texts:
    #                 f.write(passage.replace('\n', ' ') + '\n')

    def load_feature_map_by_index(self, index):
        """특정 인덱스의 feature_map 파일 하나만 로드"""
        path = os.path.join(self.feature_map_dir, f'feature_map_{index}.pt')
        if os.path.exists(path):
            return torch.load(path)
        else:
            raise FileNotFoundError(f"Feature map file not found: {path}")

    def get_num_feature_map_files(self):
        """저장된 feature_map 파일 수 반환"""
        return len([f for f in os.listdir(self.feature_map_dir) if f.startswith("feature_map_") and f.endswith(".pt")])


    # def load_all_bm25_texts(self):
    #     texts = []
    #     for fname in sorted(os.listdir(self.bm25_dir)):
    #         if fname.endswith('.txt'):
    #             with open(os.path.join(self.bm25_dir, fname), 'r', encoding='utf-8') as f:
    #                 texts.extend([line.strip() for line in f])
    #     return texts

    # def load_bm25(self):
    #     texts = self.load_all_bm25_texts()
    #     return BM25Okapi(texts)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FeatureMapManager(
        context_encoder=DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device),
        context_tokenizer=DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        documents_PATH='data/docs/psgs_w100.tsv'
    ).create_feature_maps()
if __name__ == "__main__":
    main()