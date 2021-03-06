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

DATA_ROOT = join(ret, "CommonVoice", "cv-corpus-6.1-2020-12-11", "ja")
#print("DATA_ROOT:", DATA_ROOT)

#　データ保存用ディレクトリ 
if "SAVE_ROOT" in environ:
    ret = Path(environ["SAVE_ROOT"])
else:
    ret = Path("./results")

ret = ret.expanduser()
SAVE_ROOT = join(ret, "CommonVoice", "cv-corpus-6.1-2020-12-11", "ja")
#print("SAVE_ROOT", SAVE_ROOT)


import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torchvision import transforms

torchaudio.set_audio_backend('sox_io')  # sox_io: Linux, Mac

class SpeechDataset(Dataset):
    fs = 16000

    def __init__(self, data_dir, train=True, transform=None, split_rate=0.8):
        tsv = join(data_dir, "validated.tsv")
        # データセットの一意性確認と正解ラベルの列挙
        import pandas as pd
        df = pd.read_table(tsv)
        assert not df.path.duplicated().any()
        # duplicated():重複した行を抽出    numpy.any():どれかがTrueだったらTrue
        # df.path.duplicated().any():df内のpathに重複があったらTrue
        # このpathはtsv内の要素？を指定している
        self.classes = df.client_id.drop_duplicates().tolist()
        # drop_duplicated():重複した行を削除
        # client_idの重複を削除
        self.n_classes = len(self.classes)

        # データセットの準備
        self.transform = transform
        data_dirs = tsv.split('/')
        # '/'で区切ったものをリストに
        dataset = torchaudio.datasets.COMMONVOICE('/'.join(data_dirs[:-1]),
                    tsv=data_dirs[-1])
        # 引数のurlとversionは非推奨になった

        # データセットの分割
        n_train = int(len(dataset) * split_rate)
        n_val = len(dataset) - n_train
        torch.manual_seed(torch.initial_seed())     # シードの固定
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
        self.dataset = train_dataset if train else val_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, fs, dictionary = self.dataset[idx]
        # datasetはtensor(波形, fs, tsvの要素の辞書)
        if fs != self.fs:
            x = torchaudio.transforms.Resample(fs)(x)
        # リサンプル
        # MFCC等は外部でtransformとして記述
        # ただし、推論と合わせるためにMFCCは先に済ましておく？
        x = torchaudio.transforms.MFCC(log_mels=True)(x)
        # .main()を呼び出すことに注意

        if self.transform:
            x = self.transform(x)
        # 特徴量：音声テンソル、正解ラベル：話者IDのインデックス 
        return x, self.classes.index(dictionary['client_id'])

train_dataset = SpeechDataset(DATA_ROOT, train=True)
val_test_dataset = SpeechDataset(DATA_ROOT, train=False)
# train_dataset[0][0].shape (データ数, MFCC次元, 時間サンプル)

#print(train_dataset[0][0].dim())

# 前処理の定義
Squeeze2dTo1d = lambda x: torch.squeeze(x, -3)
# lambda 引数：返り値
# squeeze 次元削除（ここではデータ数の部分が削除され、2次元の画像になっている）

""" 
音声の前処理
1. frame_period=25, hop_length=12.5で40次のMFCCに変換
2. バッチ学習では時間長を合わせる必要がある→長さを10秒（12.5ms*800）に揃える
（ほどんどの音声は10秒未満であるから対応可能、足りない場合は音声を繰り返してパディング）
3. ランダムな時間位置からランダムに2~4秒切り出す（切り出し時間位置を変えることでデータ拡張）
4. 拡大縮小して3秒に揃える（音の高低を変えずに時間軸方向を拡大縮小し、ピッチ変更によるデータの拡張）
"""

class CircularPad1dCrop:
    # 最後の1次元を指定サイズにCrop
    # 長さが足りない時はCircularPad（同じものを繰り返すパディング？）
    # 音声データの時間方向の長さを揃える
    def __init__(self, size):
        self.size = size
    def __call__(self, x):
        #print(self.size, x.size()[-1])
        n_repeat = self.size // x.size()[-1] + 1
        repeat_sizes = ((1, ) * (x.dim() - 1) + (n_repeat,))
        # 二次元形状でリピート回数を設定
        out = x.repeat(*repeat_sizes).clone()
        return out.narrow(-1, 0, self.size)
        # [tensor].narrow(dim, index, size)
        # テンソル内部のある次元のある部分を切り取って返す
        # dim:切り取る次元（行列なら、行方向で切るか(1)、列方向で切るか(2)）
        # index:切り取る部分の始点
        # size:切り取るサイズ

# 800（10秒）、320（4秒）、240（3秒）
train_transform = transforms.Compose([CircularPad1dCrop(800), transforms.RandomCrop((40, random.randint(160, 320))),
                                      transforms.Resize((40, 240)), Squeeze2dTo1d])
test_transform = transforms.Compose([CircularPad1dCrop(240), Squeeze2dTo1d])

# 学習・テストデータの準備
batch_size = 32

if train_dataset:
    train_dataset.transform = train_transform  # transformをセット
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
else:
    n_epochs = 0    # 学習データがない時は回さない

if val_test_dataset:
    val_test_dataset.transform = test_transform  # transformsをセット
    val_test_dataloader = DataLoader(val_test_dataset, batch_size=batch_size, shuffle=True)

# 学習モデル（1dCNN）
class SpeakerNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(40),
            nn.Conv1d(40, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc = nn.Sequential(
            nn.Linear(30*64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

""" 
前処理、学習、検証、推論
train_dataset：学習用データセット
val_test_dataset：検証/テスト用データセット
n_classes：分類クラス数
n_epochs：学習エポック数
load_pretrained_state：学習済みモデルを使う場合の.pthファイルパス
test_last_hidden_layer：テストデータの推論結果に最終隠れ層を使う
show_progress：エポックの学習状況をprintする
show_chart：結果をグラフ表示する
save_state：test_acc > 0.9の時のtest_loss最小更新時のstateを保存
            （load_pretrained_stateで使う）
"""

# モデルの準備
n_classes = None

if not n_classes:
    assert train_dataset, 'train_dataset or n_classes must be a valid'
    n_classes = train_dataset.n_classes

model = SpeakerNet(n_classes)

load_pretrained_state = None

if load_pretrained_state:
    model.load_state_dict(torch.load(load_pretrained_state))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 学習
n_epochs = 50
show_progress = True
save_state = False
show_chart = False
test_last_hidden_layer=False

losses = []
accs = []
val_losses = []
val_accs = []

for epoch in range(n_epochs):
    # 学習ループ
    running_loss = 0.0
    running_acc = 0.0

    for x_train, y_train in train_dataloader:
        optimizer.zero_grad()
        #print(x_train.size())
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        running_loss += loss.item()
        pred = torch.argmax(y_pred, dim=1)
        #print(pred.eq(y_train))
        running_acc += torch.mean(pred.eq(y_train).float())
        optimizer.step()
    running_loss /= len(train_dataloader)
    running_acc /= len(train_dataloader)
    losses.append(running_loss)
    accs.append(running_acc)

    # 検証ループ
    val_running_loss = 0.0
    val_running_acc = 0.0

    for val_test in val_test_dataloader:
        if not(type(val_test) is list and len(val_test) == 2):
            break
        x_val, y_val = val_test
        y_pred = model(x_val)
        val_loss = criterion(y_pred, y_val)
        val_running_loss += val_loss.item()
        pred = torch.argmax(y_pred, dim=1)
        val_running_acc += torch.mean(pred.eq(y_val).float())
        optimizer.step()
    val_running_loss /= len(val_test_dataloader)
    val_running_acc /= len(val_test_dataloader)
    can_save = (val_running_acc > 0.9 and
                val_running_loss < min(val_losses))
    val_losses.append(val_running_loss)
    val_accs.append(val_running_acc)

    if show_progress:
        print(f'epoch:{epoch}, loss:{running_loss:.3f},'
              f'acc:{running_acc:.3f}, val_loss:{val_running_loss:.3f},'
              f'val_acc:{val_running_acc:.3f}, can_save:{can_save}')
    
    # セーブ
    if save_state and can_save:
        torch.save(model.state_dict(), f'model/0001-epoch{epoch:02}.pth')
        # f''：format、{引数:02}だと前に引数に0を2つつけたものが入る

    # グラフ
    if n_epochs > 0 and show_chart:
        fig, ax = plt.subplots(2)
        # plt.subplots:figure()+add.subplotのようなもの

        ax[0].plot(losses, label='train loss')
        ax[0].plot(val_losses, label='val loss')
        ax[0].legend()  # 凡例
        ax[1].plot(losses, label='train loss')
        ax[1].plot(val_losses, label='val loss')
        ax[1].legend()
        plt.show()

    # 推論
    if not val_test_dataset:
        break
    if test_last_hidden_layer:
        model.fc = model.fc[:-1]
    
    y_preds = torch.Tensor()
    
    for val_test in val_test_dataloader:
        x_test = val_test[0] if type(val_test) is list else val_test
        y_pred = model.eval()(x_test)
        if not test_last_hidden_layer:
            y_pred = torch.argmax(y_pred, dim=1)
        y_preds = torch.cat([y_preds, y_pred])

        y_preds.detach()

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

""" from net import SpeachEmbedder, MyRNN
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


""" 
やってること
'./rensyu/data/jvs_ver1/'の下にある.DS_Store以外のフォルダ名をリスト化
後ろに'/parallel100/wav24kHz16bit'をくっつけてwavの読み込み
speaker_idxは話者ラベルに使用
Tensorのlistのmspecの長さを揃える
"""

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

"""




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
