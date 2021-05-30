import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeachEmbedder(nn.Module):
    def __init__(self, nmels, n_hidden, n_outputs, dvec_ndim, num_layer):
        super(SpeachEmbedder, self).__init__()
        self.lstm_stack = nn.LSTM(nmels, n_hidden, num_layers=num_layer,
                        batch_first=True)
        for name, param in self.lstm_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.extractor = nn.Linear(n_hidden, dvec_ndim)
        self.out = nn.Linear(n_hidden, n_outputs)

    def forward(self, x, phase, h=None):
        x, _ = self.lstm_stack(x.float())
        x = x[:,x.size(1)-1]    # 最後のフレームのみ使用
        x = nn.relu(self.extractor(x))  

        if phase == "train" or phase == "validation" or phase == "test":
            x = nn.relu(self.out(x.float())) # one-hot
        elif phase == "extract":
            x = x / torch.norm(x, dim=1).unsqueeze(1)   # d-vector
        
        return x


class MyRNN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, dvec_ndim=5, num_layers=1, bidirectional=False):
        super(MyRNN, self).__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.n_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(n_in, n_hidden, num_layers, 
                            bidirectional=bidirectional, batch_first=True)
        self.extractor = nn.Linear(self.n_direction*self.n_hidden, dvec_ndim)
        self.linear_out = nn.Linear(dvec_ndim, n_out)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers*self.n_direction, batch_size,
                        self.n_hidden)
        c0 = torch.zeros(self.num_layers*self.n_direction, batch_size,
                        self.n_hidden)
        return h0, c0

    def pack_padded(self, x, length):
        length = torch.flatten(length)
        pack = nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True)
        norm_data = (pack.data - pack.data.mean()) / pack.data.std()
        pack.data[:] = norm_data
        return pack

    def forward(self, input, lengths, phase):
        # pack_padded
        """ lengths = torch.flatten(lengths)
        print(input.shape)
        input = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        norm_data = (input.data - input.data.mean()) / input.data.std()
        input.data[:] = norm_data """
        input = self.pack_padded(input, lengths)    # class内メソッドはselfをつける
        #import pdb; pdb.set_trace()
        #print('input:', input)
        output, (h, c) = self.lstm(input)
        print('h:', h)
        #output, output_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        #output = output[:, output.size(1)-1]
        #print(output)
        h = self.extractor(h.view(-1, self.n_hidden))

        if phase=="train" or phase=="validation" or phase=="test":
            h = self.linear_out(h)
            #print(h)
        elif phase=="extract":
            h = h / torch.norm(h, dim=1).unsqueeze(1)

        return h



