import os
import torch
import pandas as pd
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from rank_bm25 import BM25Okapi

class FeatureMapManager:
    def __init__(self, context_encoder, context_tokenizer, documents_PATH, batch_size=32, tensor_path='data/tensor'):
        """
        FeatureMapManager 초기화:
        - DPR 문서 임베딩 및 BM25용 텍스트를 배치 단위로 처리하고 저장하는 클래스
        - 저장 디렉토리 생성 포함
        """
        self.context_encoder = context_encoder
        self.context_tokenizer = context_tokenizer
        self.documents_PATH = documents_PATH
        self.batch_size = batch_size
        self.tensor_path = tensor_path
        self.feature_map_dir = os.path.join(tensor_path, 'feature_maps')  # DPR 임베딩 저장 경로
        self.bm25_dir = os.path.join(tensor_path, 'bm25_passages')        # BM25 텍스트 저장 경로

        os.makedirs(self.feature_map_dir, exist_ok=True)
        os.makedirs(self.bm25_dir, exist_ok=True)

    def load_passages_in_chunks(self):
        """
        문서 전체를 배치 단위로 나누어 chunk generator로 반환
        """
        return pd.read_csv(self.documents_PATH, sep='\t', header=0, dtype={'id': str}, chunksize=self.batch_size)

    def create_feature_maps(self):
        """
        DPR context encoder를 사용하여 문서 배치 단위로 임베딩 생성 후 파일로 저장
        저장 형태: data/tensor/feature_maps/feature_map_{i}.pt
        """
        for i, chunk in enumerate(self.load_passages_in_chunks()):
            texts = chunk['text'].tolist()  # 문서 텍스트만 추출
            inputs = self.context_tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: v.to(self.context_encoder.device) for k, v in inputs.items()}
            with torch.no_grad():
                embeddings = self.context_encoder(**inputs).pooler_output.cpu()
            torch.save(embeddings, os.path.join(self.feature_map_dir, f'feature_map_{i}.pt'))

    def create_bm25_passages(self):
        """
        BM25용 텍스트 파일을 배치 단위로 저장
        저장 형태: data/tensor/bm25_passages/bm25_passages_{i}.txt
        """
        for i, chunk in enumerate(self.load_passages_in_chunks()):
            texts = chunk['text'].tolist()
            with open(os.path.join(self.bm25_dir, f'bm25_passages_{i}.txt'), 'w', encoding='utf-8') as f:
                for passage in texts:
                    f.write(passage.replace('\n', ' ') + '\n')  # 개행 제거 후 저장

    def load_all_feature_maps(self):
        """
        저장된 feature_map_{i}.pt 파일을 모두 불러와 하나의 tensor로 합침
        """
        feature_maps = []
        for fname in sorted(os.listdir(self.feature_map_dir)):
            if fname.endswith('.pt'):
                feature_maps.append(torch.load(os.path.join(self.feature_map_dir, fname)))
        return torch.cat(feature_maps, dim=0)

    def load_all_bm25_texts(self):
        """
        저장된 bm25_passages_{i}.txt 파일을 모두 읽어 하나의 리스트로 반환
        """
        texts = []
        for fname in sorted(os.listdir(self.bm25_dir)):
            if fname.endswith('.txt'):
                with open(os.path.join(self.bm25_dir, fname), 'r', encoding='utf-8') as f:
                    texts.extend([line.strip() for line in f])
        return texts

    def load_bm25(self):
        """
        BM25Okapi 객체를 생성하여 반환
        (텍스트는 배치 단위 저장 파일에서 모두 불러옴)
        """
        texts = self.load_all_bm25_texts()
        return BM25Okapi(texts)
