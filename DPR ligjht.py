from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import faiss
import numpy as np
from pathlib import Path
import pandas as pd

def load_passages(passages_file: Path, max_rows=50):  # 최대 50개만 로드하여 속도 최적화
    df = pd.read_csv(passages_file, sep='\t', header=0, dtype={'id': str}, nrows=max_rows)
    return [
        {"id": row['id'], "text": row['text'], "title": row['title']}
        for _, row in df.iterrows()
    ]

# 패시지 로드
passages = load_passages(Path("psgs_w100.tsv"))

# 컨텍스트 인코더 및 토크나이저 로드 (strict=False 추가하여 경고 방지)
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# 패시지 임베딩 생성
passage_texts = [passage["text"] for passage in passages]
passage_inputs = context_tokenizer(passage_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
passage_embeddings = context_encoder(**passage_inputs).pooler_output.detach().cpu().numpy()

# FAISS 인덱스 초기화 (내적 유사도 사용)
dimension = passage_embeddings.shape[1]
passage_index = faiss.IndexFlatIP(dimension)
passage_index.add(passage_embeddings)

# 질문 인코더 및 토크나이저 로드
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# 모델 가중치 로드 (strict=False 사용하여 경고 방지)
state_dict = context_encoder.state_dict()
context_encoder.load_state_dict(state_dict, strict=False)

# 저장된 질문 및 임베딩 리스트
past_questions = []
past_question_embeddings = []

def encode_question(query: str):
    """질문을 인코딩하여 임베딩을 반환"""
    query_inputs = question_tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
    query_embedding = question_encoder(**query_inputs).pooler_output.detach().cpu().numpy()
    return query_embedding

def encode_question_store(query: str):
    """질문을 저장하고 임베딩을 생성 (중복 저장 방지)"""
    if query not in past_questions:
        query_embedding = encode_question(query)
        past_questions.append(query)
        past_question_embeddings.append(query_embedding)
        return query_embedding
    return encode_question(query)

def get_main_referencees(query_embedding, top_k=2):  # 검색 개수 제한하여 속도 최적화
    """질문과 가장 유사한 k개의 패시지를 검색 (내적 유사도 기반)"""
    distances, indices = passage_index.search(query_embedding, top_k)
    return [(passages[i], distances[0][j]) for j, i in enumerate(indices[0])]

def get_similar_questions(new_query_embedding, top_n=2):  # 유사 질문 검색 개수 제한
    """저장된 질문과 비교하여 유클리드 거리 기반으로 가장 유사한 질문 찾기"""
    if len(past_question_embeddings) == 0:
        return []
    stored_embeddings = np.vstack(past_question_embeddings)
    distances = np.sum((stored_embeddings - new_query_embedding) ** 2, axis=1)
    closest_indices = np.argsort(distances)[:top_n]
    return [(past_questions[i], distances[i]) for i in closest_indices]

def get_sub_references(similar_questions, top_n=2):
    """유사한 질문들의 패시지를 검색하여 반환"""
    related_passages = []
    for question, _ in similar_questions:
        question_embedding = encode_question(question)
        passages = get_main_referencees(question_embedding, top_n)
        related_passages.extend(passages)
    return related_passages

def llm(query, docs1, docs2):
    """LLM 모델을 호출하여 검색된 문서들을 입력으로 제공"""
    query_embedding = encode_question_store(query)
    retrieved_passages = get_main_referencees(query_embedding, top_k=2)
    similar_questions = find_similar_questions(query_embedding, top_n=2)
    related_passages = get_sub_references(similar_questions, top_n=2)
    docs1.extend([passage['text'] for passage, _ in retrieved_passages])
    docs2.extend([passage['text'] for passage, _ in related_passages])
    return docs1, docs2


def main():
    # 테스트 쿼리 실행 (최대 3개 제한하여 속도 최적화)
    test_queries = [
        "the Hiyamas reaction was very cool",
        "How does photosynthesis work?",
        "What is the capital of France?"
    ]

    for query in test_queries:
        docs1, docs2 = llm(query, [], [])
        print(f"\nQuery: {query}")
        print("\nLLM에 전달될 문서 리스트 1:")
        for doc in docs1:
            print(doc)
        print("\nLLM에 전달될 문서 리스트 2:")
        for doc in docs2:
            print(doc)
