import pandas as pd
import json
from datasets import load_dataset

def load_passages_in_chunks(passages_file, chunk_size):
    chunks = pd.read_csv(passages_file, sep='\t', header=0, dtype={'id': str}, chunksize=chunk_size)

    for chunk in chunks:
        passages = [
            {"id": row['id'], "text": row['text'], "title": row['title']}
            for _, row in chunk.iterrows()
        ]
        yield passages  # 제너레이터로 반환, 한 번에 chunk_size개씩 처리 가능


def read_jsonl_to_list(file_path):
    # 파일에서 JSON 데이터 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data
class PassageLoader:
    def __init__(self, passages_file):
        self.passages_file = passages_file
        self.current_position = 0

    def load_passages(self, num_rows):
        df = pd.read_csv(self.passages_file, sep='\t', header=0, dtype={'id': str}, skiprows=self.current_position, nrows=num_rows)
        self.current_position += num_rows
        return [
            {"id": row['id'], "text": row['text'], "title": row['title']}
            for _, row in df.iterrows()
        ]

if __name__ == '__main__':
    path = 'data/nq/nq-dev.json'
    read_jsonl_to_list(path)