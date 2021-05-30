import numpy as np
from glob import glob
import os
from pathlib import Path
import librosa
from librosa import load
from librosa.feature import melspectrogram
from scipy import signal
from scipy.io import loadmat, savemat, wavfile
from natsort import natsorted   # 数字順にsort
import torch
import torch.nn as nn
from params import *

parser = get_params()
args = parser.parse_args()


# あとで調べる
def make_stft_args(frame_period=args.frame_period, fs=args.fs, nperseg=args.nperseg,
                    window=args.window, **kwargs):
    nshift = fs * frame_period // 1000

    if nperseg is None:
        nperseg = nshift * 4

    noverlap = nperseg - nshift

    dct = dict(window=window, nperseg=nperseg, noverlap=noverlap)

    dct["fs"] = fs
    return dct

def WavLoad(wav_path, fs=16000, fmin=0, fmax=7600, nmels=80, **kwargs):
    """ 
    入力pathからwavを読み込んでメルスペクトログラムを抽出し、正規化
    メルスペクトログラムは(時間, mel)
    overlapやhop_lengthなどの細かい設定が未実装
     """
    
    wav, fs = load(wav_path, sr=fs)

    wav = wav/np.max(np.abs(wav))
    wav *= 0.99

    _, t, Zxx = signal.stft(wav, **stft_args)
    pspec = np.abs(Zxx)
    mspec = melspectrogram(sr=fs, S=pspec, n_mels=nmels, fmin=fmin, fmax=fmax)

    return wav, fs, pspec.T, mspec.T
# 転置させるとpspec, mspecは(時間, spec)

def LoadDataName(data_path):
    """ ディレクトリ内のファイルの名称のリストを作成 """
    dataset_root = Path(data_path).expanduser()

    dirs = []

    for d in sorted(dataset_root.glob('*')):
        dirs.append(str(d))

    return list(dirs)


def MakeMspec(path, dir_speaker, train_dir, save_dir):
    """ ディレクトリ内のwavからmel spectrogram等を作成 
        mel spectrogramをnpy形式で保存"""

    wavs = []
    pspecs = []
    mspecs = []
    mspec_len = []
    speaker_label = np.empty(1)

    speaker_idx = 0

    for speaker in dir_speaker:
        load_dirs = LoadDataName(path + speaker + train_dir)
        save_dirs = path + speaker + save_dir
        print(path + speaker + train_dir)

        if os.path.exists(save_dirs):
            continue
        else:
            for idx, l_d in enumerate(load_dirs):

                wav, fs, pspec, mspec = WavLoad(l_d)
                wavs.append(wav)
                pspecs.append(pspec)
                mspecs.append(mspec)
                mspec_len.append(len(mspec))
                speaker_label = np.append(speaker_label, speaker_idx)
                os.makedirs(save_dirs, exist_ok=True)
                np.save(save_dirs+'/'+str(idx)+'mspec', mspec)
            
        speaker_idx += 1

    # return wavs, pspecs, mspecs, mspec_len, speaker_label そもそもnpyを呼び出す時にlabel等も作るから必要ない？

def LoadMspec(path, dir_speaker, save_dir):
    """ MakeMspecで保存されたmel spectrogramを読み込む"""

    mspecs = []
    mspec_len = []
    speaker_label = []
    

    speaker_idx = 0

    for idx, speaker in enumerate(dir_speaker):
        #target_vector = np.zeros(len(dir_speaker))
        #target_vector[idx] = 1

        load_dirs = natsorted(LoadDataName(path + speaker + save_dir))
        print(path + speaker + save_dir)

        for l_d in load_dirs:
            mspec = np.load(l_d, allow_pickle=True)
            #mspec = mspec.flatten()    pack_paddedに平坦化は必要なかった
            mspecs.append(torch.from_numpy(mspec))
            mspec_len.append(mspec.shape[0])
            speaker_label = np.append(speaker_label, speaker_idx)
            #speaker_label.append(target_vector)
            #import pdb; pdb.set_trace()
            
        speaker_idx += 1

    return mspecs, mspec_len, speaker_label



stft_args = make_stft_args()

path = './rensyu/data/jvs_ver1/'
train_dir = '/parallel100/wav24kHz16bit' 
save_dir = '/parallel100/mspec_train'

dir_speaker = [filename for filename in natsorted(os.listdir(path)) if not filename.startswith('.')]
speaker_classes = len(dir_speaker)

MakeMspec(path, dir_speaker, train_dir, save_dir)
# mspecs, mspecs_len, speaker_label = LoadMspec(path, dir_speaker, save_dir)



"""
その他に仕様変更したいこと
できれば辞書形式？で話者ごとに格納したい """