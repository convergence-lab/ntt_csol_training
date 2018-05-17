import sys
import os, os.path
import json
import pandas as pd

from janome.tokenizer import Tokenizer

import whoosh.index as index
from whoosh.qparser.default import QueryParser

t = Tokenizer(mmap=True)

def search(query):
    q = []
    for token in t.tokenize(query):
        if "名詞" in token.part_of_speech:
            q += [token.base_form]

    ix = index.open_dir("indexdir")
    for target in ['title', 'question', 'answer']:
        parser = QueryParser('title', ix.schema)
        q = " OR ".join(q)
        q = parser.parse(q)
        with ix.searcher() as s:
            results = s.search(q)

            return [result.values() for result in results]

def split_data():
    df = pd.read_csv('okwave.csv')
    train_df = df[:8000]
    valid_df = df[8000:9000]
    test_df  = df[9000:]

    return train_df, valid_df, test_df

def create_data(df):
    data = []
    for key, row in df.iterrows():
        print(key)
        try:
            entry = {}
            title = row.title
            title = title.replace("| 【OKWAVE】", "").replace("-", "").replace("　", "").replace(" ", "")
            entry['title'] = title
            results = search(title)
            inputs = []
            label = []
            for elem in results:
                inputs += [elem[0]]
                if elem[2] == row.title:
                    label += [1]
                else:
                    label += [0]
            entry['inputs'] = inputs
            entry['labels'] = label
            data += [entry]
        except Exception:
            print("exception")
            continue
    json_str = json.dumps(data, ensure_ascii=False)
    return json_str

def main():
    train_df, valid_df, test_df = split_data()
    train_json = create_data(train_df)
    valid_json = create_data(valid_df)
    test_json = create_data(test_df)
    with open("data/train.json", "w") as f:
        f.write(train_json)
    with open("data/valid.json", "w") as f:
        f.write(valid_json)
    with open("data/test.json", "w") as f:
        f.write(test_json)

if __name__ == '__main__':
    main()
