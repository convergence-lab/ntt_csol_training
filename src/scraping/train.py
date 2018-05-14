import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn, rnn

from model import Net

def train():
    net = Net(100, 1000)
    
if __name__ == '__main__':
    train()
