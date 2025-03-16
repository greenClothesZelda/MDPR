from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

import llm
import ref

r = None

def set_ref(question_encoder, question_tokenizer, context_encoder, context_tokenizer, documents_PATH):
    global r
    r = ref.Reference(
        context_encoder=context_encoder,
        context_tokenizer=context_tokenizer,
        question_encoder=question_encoder,
        question_tokenizer=question_tokenizer,
        documents_PATH=documents_PATH
    )

def main(question):
    main_reference, sub_reference = r.get_reference(question, 3)

    # ✅ 두 리스트를 합쳐 하나의 리스트로 저장
    all_references = main_reference + sub_reference  # ✅ 인덱스 리스트

    # ✅ passage 인덱스를 원본 텍스트로 변환하여 리스트 반환
    result_texts = [r.passage_texts[idx] for idx in all_references]

    return result_texts

if __name__ == "__main__":
    set_ref(
        context_encoder=DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        context_tokenizer=DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        question_encoder=DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
        question_tokenizer=DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
        documents_PATH='data/docs/psgs_w100.tsv'
    )

    print(main('Hello? my nfdasdfasafame is jinsu'))
    print(main('Hello? myasfdsa name is jinsu'))
    print(main('Hello? my dsafsdafname is jinsu'))
    print(main('Hello? my nsdfadsaame is jinsu'))
    print(main('Hello? mydfasfa name is jinsu'))
    r.save_Q_past()
