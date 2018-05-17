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
    df1 = pd.read_csv("syogai1.csv")
    df2 = pd.read_csv("syogai2.csv")
    df = df1.append(df2)
    df = df.loc[:, ['説明', '題名', '障害原因／対応方法']].dropna()
    sentence = []
    for key, row in df.iterrows():
        sentence += [tokenize(row['説明'])]
        sentence += [tokenize(row['題名'])]
        sentence += [tokenize(row['障害原因／対応方法'])]
    model = word2vec.Word2Vec(sentence,
                      size=50,
                      min_count=5,
                      window=5,
                      iter=10,
                      workers=4)
    model.save("word2vec.gensim.model")

if __name__ == '__main__':
    train()
