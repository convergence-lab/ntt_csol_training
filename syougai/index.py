import pandas as pd
from whoosh.fields import Schema, ID, STORED, NGRAM
from whoosh.filedb.filestore import FileStorage

df1 = pd.read_csv("syogai1.csv")
df2 = pd.read_csv('syogai2.csv')
df = df1.append(df2)
df  = df.loc[:, ['説明', '題名', '障害原因／対応方法']].dropna()
setsumei = list(df['説明'])
daimei = list(df['題名'])
geNin = list(df['障害原因／対応方法'])
schema = Schema(daimei=NGRAM(stored=True),
                setsumei=NGRAM(stored=True),
                genin=NGRAM(stored=True))
storage = FileStorage("indexdir")
ix = storage.create_index(schema)
storage.open_index()
writer = ix.writer()

for dai, setsu, geN in zip(daimei, setsumei, geNin):
    writer.add_document(daimei=dai, setsumei=setsu, genin=geN)
writer.commit(optimize=True)
