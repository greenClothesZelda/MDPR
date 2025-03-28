from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

import IO
import result
import torch
import os
import shutil


def get_second_sentence(x):
    # 문장을 구분하는 구두점 기준으로 문자열을 분리
    sentences = x.split('. ')

    # 두 번째 문장이 있는지 확인
    if len(sentences) > 1:
        return sentences[1]
    else:
        return sentences[0]


def main():


    torch.cuda.empty_cache()
    eval_data_path = 'data/nq/nq-dev.json'
    docs_path = 'data/docs/psgs_w100.tsv'

    result.set_ref(
        context_encoder=DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        context_tokenizer=DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        question_encoder=DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
        question_tokenizer=DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
        documents_PATH='data/docs/psgs_w100.tsv'
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedded_query = torch.load('data/nq/question_tensor.pt', map_location=device)
    embedded_query = embedded_query[:2]
    #print('evaldata',eval_data)
    score_list = []
    torch.save(result.main(embedded_query), 'test.pt')

if __name__ == '__main__':
    main()