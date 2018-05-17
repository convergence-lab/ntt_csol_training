import sys
import os, os.path

from janome.tokenizer import Tokenizer

import whoosh.index as index
from whoosh.qparser.default import MultifieldParser

#t = Tokenizer(mmap=True)

def search(query):
#    q = []
#    for token in t.tokenize(query):
#        q += [token.base_form]

    ix = index.open_dir("indexdir")

    parser = MultifieldParser(['title',
                               'question',
                               'answer'], ix.schema)
    q = parser.parse(query)
    with ix.searcher() as s:
        results = s.search(q)

        return [result.values() for result in results]

if __name__ == '__main__':
    results = search(sys.argv[1])
    for result in results:
        print(result[1])
