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
    for target in ['daimei', 'setsumei', 'genin']:
        parser = QueryParser('setsumei', ix.schema)
        q = " OR ".join(q)
        q = parser.parse(q)
        with ix.searcher() as s:
            results = s.search(q)

            return [result.values() for result in results]

def split_data():
    df1 = pd.read_csv('syogai1.csv')
    df2 = pd.read_csv('syogai2.csv')
    train_df = df1
    valid_df = df2
    test_df  = df2

    return train_df, valid_df, test_df

def create_data(df):
    data = []
    df = df.loc[:, ['説明', '障害原因／対応方法']].dropna()
    for key, row in df.iterrows():
        print(key)
        try:
            entry = {}
            setsumei = row['説明']
            setsumei = setsumei.replace("　", "").replace(" ", "")
            entry['setsumei'] = setsumei
            results = search(setsumei)
            inputs = []
            label = []
            for elem in results:
                inputs += [elem[0]]
                if elem[2] == row['説明']:
                    label += [1]
                else:
                    label += [0]
            entry['genin'] = inputs
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
