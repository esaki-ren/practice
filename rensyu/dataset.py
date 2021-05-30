import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, lengths, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label
        self.length = lengths

    def __len__(self):
          return self.data_num

    def __getitem__(self, idx):
        if self.transform:
          out_data = self.transform(self.data)[0][idx]
          #out_label = torch.tensor(self.label[idx])
          out_label = self.label[idx]
        else:
          out_data = self.data[idx]
          #out_label =  torch.tensor(self.label[idx])
          out_label = self.label[idx]
        length = torch.LongTensor(self.length[idx])
        return out_data, out_label, length


class MyDataset2(torch.utils.data.Dataset):
    def __init__(self, datas, labels, lengths):
        self.datas = datas
        self.labels = labels
        if isinstance(lengths, list):
          lengths = np.array(lengths)[:, None]
        elif isinstance(lengths, np.ndarray):
          lengths = lengths[:, None]
        self.lengths = lengths

    def __getitem__(self, idx):
        data= self.datas[idx],
        label = self.labels[idx]
        length = torch.LongTensor(self.lengths[idx])
        return data, label, length

    def __len__(self):
        return len(self.datas)
