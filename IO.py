import pandas as pd
def load_passages(passages_file):
    df = pd.read_csv(passages_file, sep='\t', header=0, dtype={'id': str}, nrows=10)

    return [
        {"id": row['id'], "text": row['text'], "title": row['title']}
        for _, row in df.iterrows()
    ]

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
    path = 'data/docs/psgs_w100.tsv'
    loader = load_passages(path)
    print(loader)
