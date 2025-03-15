# 컨텍스트 인코더 및 토크나이저 로드 (strict=False 추가하여 경고 방지)
from transformers import DPRContextEncoderTokenizer, DPRContextEncoder, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import IO
import torch

class Reference:
    def __init__(self, question_encoder, question_tokenizer, context_encoder, context_tokenizer, documents_PATH):
        self.question_encoder = question_encoder
        self.question_tokenizer = question_tokenizer
        self.context_encoder = context_encoder
        self.context_tokenizer = context_tokenizer

        self.Q_past = torch.tensor([])


        self.documents_PATH = documents_PATH
        self.docs = IO.load_passages(self.documents_PATH)
        self.passage_texts = [passage["text"] for passage in self.docs]
        passage_inputs = self.context_tokenizer(self.passage_texts, padding=True, truncation=True, return_tensors="pt")
        self.passage_embeddings = self.context_encoder(**passage_inputs).pooler_output

    def get_main_reference(self, embedded_query, k):
        sim = torch.matmul(self.passage_embeddings, embedded_query.T)
        top_k_values, top_k_indices = torch.topk(sim, k, dim=0)
        return top_k_indices.squeeze().tolist()

    def get_reference(self, query, k):
        query_input = self.question_tokenizer(query, return_tensors="pt")
        embedded_query = self.question_encoder(**query_input).pooler_output
        return  self.get_main_reference(embedded_query, k), self.get_sub_references(embedded_query, k)

    def get_sub_references(self, embedded_query, k):
        if k > self.Q_past.shape[0]:
            self.Q_past = torch.cat((self.Q_past, embedded_query), dim=0)
            return []

        diff = self.Q_past - embedded_query
        self.Q_past = torch.cat((self.Q_past, embedded_query), dim=0)
        diff = torch.norm(diff, dim=1)
        top_k_values, top_k_indices = torch.topk(diff, k, dim=0)
        return top_k_indices.squeeze().tolist()


    def query_embedding(self, query):
        query_input = self.question_tokenizer(query, return_tensors="pt")
        query_embedding = self.question_encoder(**query_input).pooler_output
        return query_embedding

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