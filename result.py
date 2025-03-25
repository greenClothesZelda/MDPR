from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

import llm
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

    # 🔄 passage 텍스트를 배치 단위로 순회하며 찾아냄
    result_texts = []
    current_idx = 0
    needed_indices = set(all_references)

    for chunk in IO.load_passages_in_chunks(r.feature_manager.documents_PATH, r.batch_size):
        for i, passage in enumerate(chunk):
            global_idx = current_idx + i
            if global_idx in needed_indices:
                #print(f"[Index {global_idx}] {passage['text'][:200]}...\n")  # 🔸 앞 200자만 출력
                result_texts.append(passage["text"])
            if len(result_texts) == len(all_references):
                break
        current_idx += len(chunk)
        if len(result_texts) == len(all_references):
            break

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
