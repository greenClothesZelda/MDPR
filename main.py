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
    return main_reference+sub_reference

    # response = llm.main(question, main_reference, sub_reference)
    #
    # Q_res_list = []
    # Q_res_list.append(question, response)
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
    #print(r.QA_list)