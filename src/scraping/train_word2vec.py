import pandas as pd
from gensim.models import word2vec
from janome.tokenizer import Tokenizer

t = Tokenizer(mmap=True)

def tokenize(line):
    words = []
    for token in t.tokenize(line):
        words += [token.base_form]
    return words

def train():
    df = pd.read_csv("okwave.csv")
    sentence = []
    for key, row in df.iterrows():
        sentence += [tokenize(row.title)]
        sentence += [tokenize(row.question)]
        sentence += [tokenize(row.answer)]
    model = word2vec.Word2Vec(sentence,
                      size=100,
                      min_count=5,
                      window=5,
                      iter=10,
                      workers=4)
    model.save("word2vec.gensim.model")

if __name__ == '__main__':
    train()
