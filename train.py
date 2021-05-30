import sys
from os import environ
from os.path import dirname, join, expanduser
from pathlib import Path

# データセット読み込みディレクトリ 
if "DATASET_ROOT" in environ:
    ret  = Path(environ["DATASET_ROOT"])
else:
    ret = Path("~", "dataset")

ret = ret.expanduser()

ret.mkdir(exist_ok=True, parents=True)

DATA_ROOT = join(ret, "practice")
print("DATA_ROOT:", DATA_ROOT)

#　データ保存用ディレクトリ 
if "SAVE_ROOT" in environ:
    ret = Path(environ["SAVE_ROOT"])
else:
    ret = Path("./results")

ret = ret.expanduser()
SAVE_ROOT = join(ret, "practice")
print("SAVE_ROOT", SAVE_ROOT)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from glob import glob
from pathlib import Path
import os
import librosa
from librosa import load
from librosa.feature import melspectrogram
from scipy import signal
from scipy.io import loadmat, savemat, wavfile
from natsort import natsorted
import matplotlib.pyplot as plt

import time
from params import *

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from net import SpeachEmbedder, MyRNN
#from net_lightning import SpeachEmbedder

from make_mspec import LoadMspec
from dataset import MyDataset, MyDataset2

np.set_printoptions(threshold=10)


# あとで調べる
def make_stft_args(frame_period=5, fs=16000, nperseg=None, window='hann', **kwargs):
    nshift = fs * frame_period // 1000

    if nperseg is None:
        nperseg = nshift * 4

    noverlap = nperseg - nshift

    dct = dict(window=window, nperseg=nperseg, noverlap=noverlap)

    dct["fs"] = fs
    return dct



""" 
やってること
'./rensyu/data/jvs_ver1/'の下にある.DS_Store以外のフォルダ名をリスト化
後ろに'/parallel100/wav24kHz16bit'をくっつけてwavの読み込み
speaker_idxは話者ラベルに使用
Tensorのlistのmspecの長さを揃える
"""

parser = get_params()
args = parser.parse_args()


path = './rensyu/data/jvs_ver1/'
train_dir = '/parallel100/wav24kHz16bit' 
save_dir = '/parallel100/mspec_train'

dir_speaker = [filename for filename in natsorted(os.listdir(path)) if not filename.startswith('.')]
speaker_classes = len(dir_speaker)

mspecs, mspecs_len, speaker_label = LoadMspec(path, dir_speaker, save_dir)
#mean = torch.mean(torch.stack(mspecs))
#import pdb; pdb.set_trace()


num_layers = 1
n_hidden = 80
batch_size = 10
n_epoch = 3
lr = 0.001
weight_decay = 1e-6
model_save_path = './rensyu/tutorial/model_param.pt'
use_cuda = torch.cuda.is_available()

def train(model, optimizer, datas, labels, lengths, save_dir):
    if use_cuda:
        model = model.cuda()

    datas = nn.utils.rnn.pad_sequence(datas, batch_first=True)
    labels = torch.LongTensor(labels).flatten()

    dataset = MyDataset2(datas, labels, lengths)


    train_rate = 0.8
    train_size = int(len(dataset) * train_rate)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    #print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataset_loaders = {"train": train_loader, "validation": val_loader}

    # training loop
    criterion = nn.CrossEntropyLoss()

    loss_history = {"train": [], "validation": []}
    print("Start Training...")

    start_time = time.time()

    for epoch in range(1, n_epoch+1):
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            elif phase == "validation":
                model.eval()
            
            running_loss = 0
            print('phase:', phase)
            for data, label, lengths in dataset_loaders[phase]:
                episode_start_time = time.time()

                # ここまでは合ってる

                # sorted
                #sorted_length, sorted_idx = torch.sort(lengths.view(-1), dim=0, descending=True)
                sorted_length, sorted_idx = lengths.sort(0,descending=True)
                #sorted_length = sorted_length.long().numpy()
                data = torch.squeeze(torch.stack(data, 1), 1) # dataloaderでlistにされてるからここでstack
                # stackだと重ねた次元は減らないからsqueeze

                sorted_label = label[sorted_idx]
                sorted_data = data[sorted_idx]
                sorted_data = torch.squeeze(sorted_data, 1)

                #import pdb; pdb.set_trace()

                #h, c = model.init_hidden(len(sorted_length))
                
                if use_cuda:
                    sorted_data, sorted_label = sorted_data.cuda(), sorted_label.cuda()
                    #h, c = h.cuda(), c.cuda()
                
                optimizer.zero_grad()

                

                output = model(sorted_data, sorted_length, phase)


                loss = criterion(output, label)
                
                
                if phase == "train":
                    print('---backward---')
                    loss.backward()
                    print('---optim---')
                    optimizer.step()
                running_loss += loss.data.item()
                print('Episode time: %1.3f   Episode Loss: %1.3f'  %(time.time() - episode_start_time, loss.item()))
            loss_history[phase].append(running_loss / (len(dataset_loaders[phase])))
            print('loss_history', loss_history[phase])

            fig = plt.figure()
            plt.plot(loss_history[phase])
            plt.title('train_loss_history(phase:{0}, epoch:{1}'.format(phase,epoch))
            plt.ylabel("loss")
            plt.grid()

            fig.savefig('./rensyu/tutorial/train_loss_history({0}).png'.format(phase))

        torch.save(model.state_dict(), save_dir)

    return loss_history

model = MyRNN(n_in=80, n_hidden=n_hidden, n_out=speaker_classes)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if os.path.isfile(model_save_path):
    model.load_state_dict(torch.load(model_save_path))

loss_history = train(model=model, optimizer=optimizer, datas=mspecs, 
                        labels=speaker_label, lengths=mspecs_len, save_dir=model_save_path)

torch.save(model.state_dict(), model_save_path)






""" lstm = nn.RNN(80, 5, batch_first=True)

packed_out, ht = lstm(mspecs_packed)

out, _ = pad_packed_sequence(packed_out)

print('output:',out)
print ('only last:',ht[-1])
 """


""" model = SpeachEmbedder(80, 16, speaker_classes, 2)
trainer = pl.Trainer(max_epochs=3)
trainer.fit(model, loader_train)

model_save_path = './rensyu/tutrial/model_param.pt'

torch.save(model.state_dict(), model_save_path)
 """
""" モデルのロード
model = SpeachEmbedder(80, 32, speaker_classes, 2)
model.load_state_dict(torch.load(model_save_path)) """



""" mspecs = nn.utils.rnn.pad_sequence(mspecs, batch_first=True)
mspecs (音声の数, 時間サンプル, mspecの値)
speaker_label = np.array(speaker_label, dtype='int')
speaker_label = speaker_label.reshape(-1,1)
"""


""" model = SpeachEmbedder(80, 16, speaker_classes, 5)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print('Train Start')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(1, epochs+1):
    train(loader_train, model, optimizer, criterion, device, epochs, epoch)
 """
