import json
import torch
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
# from rank_bm25 import BM25Okapi  # 🔒 BM25 비활성화

import result

# ✅ (1) Natural Questions (NQ) 데이터셋 로드
def load_nq_dataset(file_path, num_samples=100):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions, answers, documents = [], [], []

    for item in data[:num_samples]:
        question = item["question"]
        answer = item["answers"][0] if item["answers"] else ""
        long_answer = item["positive_ctxs"][0]["text"] if item["positive_ctxs"] else ""

        questions.append(question)
        answers.append(answer)
        documents.append(long_answer)

    return questions, answers, documents

# ✅ (2) BM25 검색 모델 구축 (❌ 비활성화)
# def build_bm25_index(documents):
#     tokenized_docs = [doc.split() for doc in documents]
#     return BM25Okapi(tokenized_docs), tokenized_docs

# ✅ (3) DPR 모델 및 토크나이저 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
dpr_question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
dpr_context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)

question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# ✅ (4) DPR 임베딩 생성 함수
def encode_dpr_passages(passages, model):
    inputs = context_tokenizer(passages, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model(**inputs).pooler_output
    return embeddings

def encode_dpr_question(question, model):
    inputs = question_tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model(**inputs).pooler_output
    return embedding

# ✅ (5) Top-k 검색 및 정확도 평가 (BM25 ❌, DPR ✅, MDPR ✅)
def compute_top_k_accuracy(questions, answers, documents, dpr_doc_embeddings, k=3):
    correct_dpr, correct_mdpr = 0, 0
    total = len(questions)

    for i, question in enumerate(questions):
        # 🔹 DPR 검색
        question_embedding_dpr = encode_dpr_question(question, dpr_question_encoder)
        similarities_dpr = torch.matmul(question_embedding_dpr, dpr_doc_embeddings.T).squeeze(0)
        top_k_indices_dpr = torch.topk(similarities_dpr, k=k).indices.tolist()
        top_k_dpr = [documents[idx] for idx in top_k_indices_dpr]

        # 🔹 MDPR 검색 (사용자 정의 모델)
        top_k_mdpr = result.main(question)

        # 🔹 정답 포함 여부 확인
        if any(answers[i] in doc for doc in top_k_dpr):
            correct_dpr += 1
        if any(answers[i] in doc for doc in top_k_mdpr):
            correct_mdpr += 1

    return correct_dpr / total, correct_mdpr / total

# ✅ 실행: NQ 데이터셋을 불러와서 Top-k Retrieval Accuracy 평가
nq_file_path = "/Users/minjune/IdeaProjects/MDPR/data/nq/nq-dev.json"
questions, answers, documents = load_nq_dataset(nq_file_path)

# BM25 비활성화
# bm25, tokenized_docs = build_bm25_index(documents)

# DPR 문서 임베딩 생성
dpr_doc_embeddings = encode_dpr_passages(documents, dpr_context_encoder)

# Top-k Retrieval Accuracy 계산
top_k_accuracy_dpr, top_k_accuracy_mdpr = compute_top_k_accuracy(
    questions, answers, documents, dpr_doc_embeddings, k=5
)

# 🔽 출력
print(f"DPR Top-5 Retrieval Accuracy: {top_k_accuracy_dpr:.2%}")
print(f"MDPR Top-5 Retrieval Accuracy: {top_k_accuracy_mdpr:.2%}")
