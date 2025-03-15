import pandas as pd
def load_passages(passages_file):
    df = pd.read_csv(passages_file, sep='\t', header=0, dtype={'id': str}, nrows=10)

    return [
        {"id": row['id'], "text": row['text'], "title": row['title']}
        for _, row in df.iterrows()
    ]

if __name__ == '__main__':
    path = 'data/docs/psgs_w100.tsv'
    passages = load_passages(path)
    print(passages)
    passages = load_passages(path)
    print(passages)
