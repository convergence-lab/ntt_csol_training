from gensim.models import word2vec
from janome.tokenizer import Tokenizer

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn, rnn

class Net(gluon.Block):
    def __init__(self, rnn_units, max_length, w2v_model, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.max_length = max_length
        idx2w = list(w2v_model.wv.vocab)
        w2idx = {w: i for i, w in enumerate(idx2w)}
        w2v_weight = nd.array([w2v_model.wv[w] for i, w in enumerate(idx2w)])

        with self.name_scope():
            self.proj = nn.Dense(100)
            self.lstm = rnn.LSTM(rnn_units)
            self.pool = nn.MaxPool2D((1, max_length))
            self.attn = nn.Dense(max_length)
            self.out = nn.Dense(2)

    def forward(self, q, a):
        q = self.proj(q)
        q = self.lstm(q)
        qvec = self.pool(q)

        a = self.proj(a)
        a = self.lstm(a)

        attn = nd.batch_dot(qvec, a)
        attn = nd.softmax(self.atten(attn))

        a = nd.batch_dot(attn, a)
        avec = self.pool(a)

        out = self.out(nd.cocat(qvec, avec, q-a, q*a))
        out = nd.log_softmax(out)
        return out
