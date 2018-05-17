import sys
import json
from tqdm import tqdm
import pickle

import numpy as np
from janome.tokenizer import Tokenizer
from gensim.models import word2vec

import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pad_sequence

from model_pt import Net

from search import search

def load_data(w2v_model):
    with open('data/train.json') as f:
        train_json = json.load(f)
    with open('data/valid.json') as f:
        valid_json = json.load(f)
    return train_json, valid_json

def get_wordvector(word, w2v_model):
    try:
        vec = w2v_model.wv[word]
    except:
        vec = np.zeros(50, dtype=np.float32)
    return vec.tolist()

def parse_line(line, w2v_model, t):
    vec = []
    for token in t.tokenize(line):
        vec += [get_wordvector(token.base_form, w2v_model)]
    if len(vec) == 0:
        vec += [np.zeros(50).tolist()]
    return vec

def shuffle(data):
    state = np.random.get_state()
    np.random.shuffle(data)
    return data

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype=np.long)[y]

def predict(query):
    device = torch.device('cpu')
    max_length = 100
    t = Tokenizer(mmap=True)
    w2v_model = word2vec.Word2Vec.load("./word2vec.gensim.model")
    net = torch.load('models/net-1.pt')
    prob = []
    with torch.no_grad():
        title = query
        candidates = search(query)
        title = parse_line(title, w2v_model, t)
        if len(title) > max_length:
            title = title[:max_length]
        else:
            title = np.pad(title, [[0, max_length - len(title)], [0, 0]], 'constant')
        title = torch.Tensor([title]).to(device)
        for cand in candidates:
            cand_vec = parse_line(cand, w2v_model, t)
            if len(cand_vec) > max_length:
                cand_vec = cand_vec[:max_length]
            else:
                cand_vec = np.pad(cand_vec, [[0, max_length - len(cand_vec)], [0, 0]], 'constant')
            cand_vec = torch.Tensor([cand_vec]).to(device)
            output = net.forward(title, cand_vec)
            output = output.to(torch.device("cpu")).numpy()
            prob += [output[0]]
    if len(prob) != 0:
        ind = np.argmax(prob)
        print(candidates[ind])
    else:
        print("no hit")

if __name__ == '__main__':
    predict(sys.argv[1])
