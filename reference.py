from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch

from pathlib import Path
import pandas as pd

import llm
from IO import load_passages


def get_main_references():

def get_sub_references():

def sim_query_passages():

def query_likelyhood():



passages = load_passages(Path("data/docs/psgs_w100.tsv"))

# Load the context encoder and tokenizer
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Encode the passages
passage_texts = [passage["text"] for passage in passages]
passage_inputs = context_tokenizer(passage_texts, padding=True, truncation=True, return_tensors="pt")
passage_embeddings = context_encoder(**passage_inputs).pooler_output

# Load the question encoder and tokenizer
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# Encode the query
query = "the Hiyamas reaction was very cool"
query_inputs = question_tokenizer(query, return_tensors="pt")
query_embedding = question_encoder(**query_inputs).pooler_output

# Compute similarity scores
scores = torch.matmul(query_embedding, passage_embeddings.T)
scores = scores.squeeze().tolist()

sorted_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)

k = 3
k_passages = sorted_passages[:k]
# Print the scores
# for passage, score in sorted_passages[:k]:
# #     print(f"Passage ID: {passage['id']}, Title: {passage['title']}, Score: {score}")

docs_ori = [passage['text'] for passage, score in k_passages]

llm.main(query, docs_ori, docs_ori)

