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

    eval_data = IO.read_jsonl_to_list(eval_data_path)
    #print('evaldata',eval_data)
    score_list = []
    for item in eval_data:
        #print(item)
        question = item['question']
        docs = result.main(question)
        sum_score = 0.0
        #print('docs:', docs)

        for doc in docs:
            #print(f'{doc}, score:', end=' ')
            score = 0.0

            for positive in item['positive_ctxs']:
                #print(type(positive), positive, end=' ')
                if str(doc) in positive['passage_id']:
                    score += positive['score']

            for negative in item['hard_negative_ctxs']:
                if str(doc) in negative['passage_id']:
                    score -= negative['score']

            #print(score)
            sum_score += score
        print(question,': Total score:', sum_score)
        score_list.append(sum_score)
    print('scores', score_list)

if __name__ == '__main__':
    main()