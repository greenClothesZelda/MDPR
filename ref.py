# 컨텍스트 인코더 및 토크나이저 로드 (strict=False 추가하여 경고 방지)
from transformers import DPRContextEncoderTokenizer, DPRContextEncoder, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

import IO


class Reference:
    def __init__(self, question_encoder, question_tokenizer, context_encoder, context_tokenizer, documents_PATH):
        self.question_encoder = question_encoder
        self.question_tokenizer = question_tokenizer
        self.context_encoder = context_encoder
        self.context_tokenizer = context_tokenizer

        self.documents_PATH = documents_PATH
        self.docs = IO.load_passages(self.documents_PATH)
        self.passage_texts = [passage["text"] for passage in self.docs]
        passage_inputs = self.context_tokenizer(self.passage_texts, padding=True, truncation=True, return_tensors="pt")
        passage_embeddings = self.context_encoder(**passage_inputs).pooler_output
        self.docs['embeddings'] = passage_embeddings

    def get_reference(self, k):
        reference_list = []
        reference_list.append()
        return

    def get_main_reference(self, k):
        reference_list = []


    def query_embedding(self, query):
        query_input =  self.question_tokenizer.tokenize(query, return_tensors="pt")
        query_embedding = self.question_encoder(**query_input).pooler_output
        return query_embedding


if __name__ == "__main__":
    ref = Reference(context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
                    , context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
                    , question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
                    , question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
                    , documents_PATH='data/docs/psgs_w100.tsv')
    print(ref.docs)
