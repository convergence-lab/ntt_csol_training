import os, os.path
import pandas as pd
from whoosh.fields import Schema, ID, STORED, NGRAM
from whoosh.filedb.filestore import FileStorage


def create_index():
    schema = Schema(title=NGRAM(stored=True),
                    question=NGRAM(stored=True),
                    answer=NGRAM(stored=True))
    storage = FileStorage("indexdir")
    ix = storage.create_index(schema)
    storage.open_index()
    writer = ix.writer()

    df = pd.read_csv('okwave.csv')
    for key, row in df.iterrows():
        writer.add_document(title=row.title, question=row.question, answer=row.answer)
    writer.commit(optimize=True)

if __name__ == '__main__':
    create_index()
