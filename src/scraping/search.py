import os, os.path
from whoosh.fields import Schema, ID, STORED, TEXT
from whoosh.filedb.filestore import FileStorage
import whoosh.index as index
from whoosh.qparser.default import MultifieldParser

def search(query):
    ix = index.open_dir("indexdir")

    parser = MultifieldParser(['title',
                               'question',
                                   'answer'], ix.schema)
    q = parser.parse(query)
    with ix.searcher() as s:
        results = s.search(q)

        return [result.values() for result in results]

if __name__ == '__main__':
    results = search("ペット")
    print(results)
    for result in results:
        print(result[2])
