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

def load_data(w2v_model):
    with open('data/train.json') as f:
        train_json = json.load(f)
    with open('data/valid.json') as f:
        valid_json = json.load(f)
    # train_data = parse_data(train_json, w2v_model, t)
    # valid_data = parse_data(valid_json, w2v_model, t)
    # with open('train.pkl', "wb") as f:
        # pickle.dump(train_data, f)
    # with open('valid.pkl', "wb") as f:
        # pickle.dump(valid_data, f)
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

def batchfy(data, batch_size, max_length, w2v, t):
    batch_title = []
    batch_cand = []
    batch_label= []

    while True:
        shuffle(data)
        for i, elem in enumerate(data):
            title = elem['title']
            candidates = elem['inputs']
            labels = elem['labels']
            if len(labels) == 0:
                continue
            title_vec = parse_line(title, w2v, t)
            pos_index = np.argmax(labels, axis=0)
            for i in range(3):
                neg_index = np.random.randint(0, len(labels))
                if labels[neg_index] == 0:
                    break
            if labels[neg_index] == 1:
                continue
            title_vec = np.pad(title_vec, [[0, max_length - len(title_vec)], [0, 0]], 'constant')
            batch_title += [title_vec]
            cand = parse_line(candidates[pos_index], w2v, t)
            if len(cand) > max_length:
                cand = cand[:max_length]
            else:
                cand = np.pad(cand, [[0, max_length - len(cand)], [0, 0]], 'constant')
            batch_cand += [cand]
            batch_label += [1]

            batch_title += [title_vec]
            cand = parse_line(candidates[pos_index], w2v, t)
            if len(cand) > max_length:
                cand = cand[:max_length]
            else:
                cand = np.pad(cand, [[0, max_length - len(cand)], [0, 0]], 'constant')
            batch_cand += [cand]
            batch_label += [0]

            if len(batch_title) == batch_size:
                batch_title = torch.Tensor(batch_title)
                batch_cand = torch.Tensor(batch_cand)
                # batch_label = to_categorical(batch_label, 2)
                # batch_label = torch.from_numpy(batch_label)
                batch_label = torch.from_numpy(np.array(batch_label, dtype=np.long))
                yield batch_title, batch_cand, batch_label
                batch_title = []
                batch_cand = []
                batch_label = []

def train():
    epochs = 10
    steps_per_epoch = 100

    in_units = 50
    units = 50
    max_length = 100
    batch_size = 100
    num_layers = 1
    use_cuda = True
    t = Tokenizer(mmap=True)
    w2v_model = word2vec.Word2Vec.load("./word2vec.gensim.model")
    train_data, valid_data = load_data(w2v_model)

    generator = batchfy(train_data, batch_size, max_length, w2v_model, t)
    device = torch.device("cuda" if use_cuda else "cpu")
    net = Net(in_units, units, num_layers, max_length, batch_size).to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    for ep in range(epochs):
        train_loss = 0
        for batch_i in tqdm(range(steps_per_epoch)):
            optimizer.zero_grad()
            title, cand, label = next(generator)
            title = title.to(device)
            cand = cand.to(device)
            label = label.to(device)
            output = net.forward(title, cand)
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.sum().item()
        train_loss /= steps_per_epoch * batch_size
        print(f"Train loss: {train_loss}")
        valid_loss, acc = valid(net, valid_data, max_length, w2v_model, t, device)
        print(f"Valid loss: {valid_loss}, Acc: {acc}")

def valid(net, valid_data, max_length, w2v_model, t, device):
    valid_loss = 0
    acc = 0
    n_total = 0
    with torch.no_grad():
        for i, elem in enumerate(valid_data):
            title = elem['title']
            candidates = elem['inputs']
            labels = elem['labels']
            title = parse_line(title, w2v_model, t)
            title = np.pad(title, [[0, max_length - len(title)], [0, 0]], 'constant')
            title = torch.Tensor([title]).to(device)
            for cand, label in zip(candidates, labels):
                cand = parse_line(cand, w2v_model, t)
                if len(cand) > max_length:
                    cand = cand[:max_length]
                else:
                    cand = np.pad(cand, [[0, max_length - len(cand)], [0, 0]], 'constant')
                cand = torch.Tensor([cand]).to(device)
                dlabel = torch.from_numpy(np.array([label], dtype=np.long)).to(device)
                output = net.forward(title, cand)
                loss = F.nll_loss(output.unsqueeze(0), dlabel).sum().item()
                valid_loss += loss
                output = output.to(torch.device("cpu")).numpy()
                pred = np.argmax(output)
                acc += 1 if pred == label else 0
                n_total += 1
        loss /= n_total
        acc /= n_total
    return valid_loss, acc

if __name__ == '__main__':
    train()
