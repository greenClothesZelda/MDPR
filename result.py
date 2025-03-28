from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

# import llm
import ref
import IO

r = None

def set_ref(question_encoder, question_tokenizer, context_encoder, context_tokenizer, documents_PATH):
    global r
    if r is None:  # ✅ 한 번만 실행되도록 체크
        r = ref.Reference(
            context_encoder=context_encoder,
            context_tokenizer=context_tokenizer,
            question_encoder=question_encoder,
            question_tokenizer=question_tokenizer,
            documents_PATH=documents_PATH
        )


# ✅ `result.py`가 import될 때 `set_ref()` 자동 실행
if r is None:
    set_ref(
        context_encoder=DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        context_tokenizer=DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
        question_encoder=DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
        question_tokenizer=DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
        documents_PATH='data/docs/psgs_w100.tsv'
    )


def main(question):
    if r is None:
        raise ValueError("❌ Error: `set_ref()` must be called before using `main()`.")

    all_references = r.get_reference(question, 9)

    return all_references