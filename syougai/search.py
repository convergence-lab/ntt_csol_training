import sys
import os, os.path
from whoosh.fields import Schema, ID, STORED, TEXT
from whoosh.filedb.filestore import FileStorage
import whoosh.index as index
from whoosh.qparser.default import MultifieldParser

def search(query):
    ix = index.open_dir("indexdir")

    parser = MultifieldParser(['daimei',
                               'setsumei',
                               'genin'], ix.schema)
    q = parser.parse(query)
    with ix.searcher() as s:
        results = s.search(q)

        return [result.values()[1] for result in results]

if __name__ == '__main__':
    results = search(sys.argv[1])
    for result in results:
        print(result)
