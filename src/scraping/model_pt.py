import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_units, units, rnn_layers, max_length, batch_size):
        super(Net, self).__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        self.proj = nn.Linear(in_units, units)
        self.lstm = nn.LSTM(units, units, rnn_layers)
        self.pool = nn.MaxPool2d((max_length, 1))
        self.attn = nn.Linear(max_length, max_length)
        self.out = nn.Linear(units*4, 2)

    def forward(self, q, a):
        q = F.relu(self.proj(q))
        q, _ = self.lstm(q)
        qvec = self.pool(q)

        a = F.relu(self.proj(a))
        a, _ = self.lstm(a)

        attn = torch.bmm(qvec, torch.transpose(a, 1, 2))
        attn = F.softmax(self.attn(attn), dim=-1)
        attn = torch.transpose(attn, 1, 2)
        a = attn * a

        avec = self.pool(a)

        out = self.out(torch.cat((qvec, avec, qvec-avec, qvec*avec), dim=-1))
        out = F.log_softmax(out, dim=-1).squeeze()
        return out
