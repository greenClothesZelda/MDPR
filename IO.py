import pandas as pd
import json
from datasets import load_dataset
def load_passages(passages_file):
    df = pd.read_csv(passages_file, sep='\t', header=0, dtype={'id': str}, nrows=10)

    return [
        {"id": row['id'], "text": row['text'], "title": row['title']}
        for _, row in df.iterrows()
    ]

def read_jsonl_to_list(file_path):
    # 파일에서 JSON 데이터 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # for item in data:
    #     print(f"Dataset: {item['dataset']}")
    #     print(f"Question: {item['question']}")
    #     print("Answers:", item['answers'])
    #     print("Positive Contexts:")
    #     # 수정된 코드 (txt 키 확인)
    #     print(item['positive_ctxs'])
    #     print("Negative Contexts:")
    #     print(type(item['negative_ctxs']))
    #     print('hard_negative_ctxs')
    #     print((item['hard_negative_ctxs'][0]['score']))
    #     print("\n")
    #     break
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