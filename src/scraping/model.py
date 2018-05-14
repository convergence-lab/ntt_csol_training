from gensim.models import word2vec
from janome.tokenizer import Tokenizer

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn, rnn

class Net(gluon.Block):
    def __init__(self, rnn_units, max_length, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.max_length = max_length
        w2v_model = word2vec.Word2Vec.load("./word2vec.gensim.model")
        idx2w = list(w2v_model.keys())
        w2idx = {w: i for i, w in enumerate(idx2w)}
        w2v_weight = nd.array(w2v_model.values())

        with self.name_scope():
            self.embed = nn.Embedding(len(idx2w), 100, weight_initializer=mx.initializer.Constant(w2v_weight))
            self.lstm = rnn.LSTM(rnn_units)
            self.pool = nn.MaxPool2D((1, max_length))
            self.attn = nn.Dense(max_length)
            self.out = nn.Dense(2)

    def forward(self, q, a):
        q = self.embed(q)
        q = self.lstm(q)
        qvec = self.pool(q)

        a = self.embed(a)
        a = self.lstm(a)
        attn = nd.batch_dot(q, a)
        attn = nd.softmax(self.atten(attn))
        a = nd.batch_dot(att, a)
        avec = self.pool(a)

        out = self.out(nd.cocat(qvec, avec, q-a, q*a))
        out = nd.log_softmax(out)
        return out
