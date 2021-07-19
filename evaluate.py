import numpy as np
import librosa
import glob
from pathlib import Path
import pyreaper
import pysptk
from scipy.io import wavfile
import pyworld as pw
import matplotlib.pyplot as plt
import japanize_matplotlib
from os import environ
from pathlib import Path

# 信頼区間測定用
import pandas as pd
import scipy as sp
import seaborn as sns

from pystoi.stoi import stoi
from pesq import pesq

def get_vuv(f0):
    return (f0 > 1).astype(int)

# vuv一致率計算
def vuv_estimate(wave_A, wave_B, fs):
    _, _, f0_times_A, f0_A, _ = pyreaper.reaper(wave_A, fs)
    _, _, f0_times_B, f0_B, _ = pyreaper.reaper(wave_B, fs)

    length = min(f0_A.size, f0_B.size)
    f0_A = f0_A[:length]
    f0_B = f0_B[:length]

    vuv_A = get_vuv(f0_A)
    vuv_B = get_vuv(f0_B)

    vuv_match = (vuv_A == vuv_B).astype(int).sum() / vuv_A.size
    vuv_error = 1 - vuv_match
    vuv_match *= 100
    vuv_error *= 10
    vuv = (vuv_A * vuv_B) == 1
    f0v_A = f0_A[vuv]
    f0v_B = f0_B[vuv]

    return f0v_A, f0v_B, vuv_match, vuv_error

# 客観評価
def evaluator(path_list, data_type='mspec'):

    df_list = []

    for path in path_list:
        print(path)
        folder_list = list(sorted(glob.glob(str(path / 'ATR503_j**_0'))))#[:3] # 全フォルダ名のlist
        assert len(folder_list) > 0, path

        dct = {}
        for i in folder_list:
            name = Path(i).stem.rsplit("_", 1)[0]
            if name in dct:
                continue

            path_t = i + '/target.wav'

            if data_type=='mspec':
                path_y = i + '/teacher_forcing.wav'
            elif data_type=='world':
                path_y = i + '/multi_task(world)_tf.wav'

            fs, wave_t = wavfile.read(path_t)
            fs, wave_y = wavfile.read(path_y)

            length = min(wave_t.shape[0], wave_y.shape[0])

            wave_t = wave_t[:length]
            wave_y = wave_y[:length]

            f0v_A, f0v_B, vuv_match, vuv_error = vuv_estimate(wave_t, wave_y, fs)

            rms = np.sqrt(np.mean((f0v_A-f0v_B)**2))
            corr = np.corrcoef(f0v_A, f0v_B)[0, 1]

            stoi_score = stoi(wave_t, wave_y, fs)
            pesq_score = pesq(fs, wave_t, wave_y, 'wb')

            dct[name] = {"stoi": stoi_score, "pesq": pesq_score,
                         "vuv_match": vuv_match, "vuv_error": vuv_error, "rms": rms, "corr": corr}

        df = pd.DataFrame(dct).T
        df_list.append(df)
    
    return df_list

# 95%信頼区間
def confidence_interval(list):  # 95%信頼区間

    var = np.var(list, ddof=1)  # 不偏分散
    mean = np.mean(list)  # 標本平均
    deg_of_freedom = len(list)-1  # t分布の自由度(標本数-1)

    bottom, up = sp.stats.t.interval(
        alpha=0.95,  # 信頼区間
        loc=mean,  # 標本平均
        scale=np.sqrt(var/len(list)),  # 標準誤差(推定量の標準偏差)
        df=deg_of_freedom  # 自由度
    )

    error_bar = up - mean

    return mean, error_bar

data_root = Path('~', 'data')
data_root = data_root.expanduser()
data_root = data_root / 'lip2sp'

dir_list = ['202107121550_p79y_F01_kabulab_mspec', '202107141343_2jch_F01_kabulab_mspec', '202105101801_yqys_F01_kabulab_mspec']
condition_list = ['ATR', 'ATR&balanced', 'ALL']

eval_list = []

for dir in dir_list:
    path = data_root / dir / 'results/F01_kabulab'
    eval_list.append(path)

df_list_mspec = evaluator(eval_list, 'mspec')
df_list_world = evaluator(eval_list, 'world')

df_list = [df_list_mspec, df_list_world]
dtype_list = ['mspec', 'world']


for list, dtype in zip(df_list, dtype_list):

    pesq_dict = {'mean': [], 'error': []}
    stoi_dict = {'mean': [], 'error': []}
    vuv_match_dict = {'mean': [], 'error': []}
    vuv_error_dict = {'mean': [], 'error': []}
    rms_dict = {'mean': [], 'error': []}
    corr_dict = {'mean': [], 'error': []}

    for df_dict in list:

        stoi = df_dict["stoi"].values
        pesq = df_dict["pesq"].values
        vuv_match = df_dict["vuv_match"].values
        vuv_error = df_dict["vuv_error"].values
        rms = df_dict["rms"].values
        corr = df_dict["corr"].values

        pesq_mean, pesq_error = confidence_interval(pesq)
        stoi_mean, stoi_error = confidence_interval(stoi)
        vuv_match_mean, vuv_match_error = confidence_interval(vuv_match)
        vuv_error_mean, vuv_error_error = confidence_interval(vuv_error)
        rms_mean, rms_error = confidence_interval(rms)
        corr_mean, corr_error = confidence_interval(corr)

        pesq_dict['mean'].append(pesq_mean)
        pesq_dict['error'].append(pesq_error)

        stoi_dict['mean'].append(stoi_mean)
        stoi_dict['error'].append(stoi_error)

        vuv_match_dict['mean'].append(vuv_match_mean)
        vuv_match_dict['error'].append(vuv_match_error)

        vuv_error_dict['mean'].append(vuv_error_mean)
        vuv_error_dict['error'].append(vuv_error_error)

        rms_dict['mean'].append(rms_mean)
        rms_dict['error'].append(rms_error)

        corr_dict['mean'].append(corr_mean)
        corr_dict['error'].append(corr_error)

    print(pesq_dict, stoi_dict, vuv_error_dict, rms_dict, corr_dict)

    x_axis = condition_list
    colorlist = ["y", "g", "b"]


    # PESQのグラフ
    fig = plt.figure()
    plt.bar(x_axis, pesq_dict['mean'],
            yerr=pesq_dict['error'], width=0.5, capsize=4, color=colorlist)
    plt.title('PESQ'+'('+dtype+')', fontsize=16)
    plt.xlabel('学習に使用したデータ数', fontsize=14)
    plt.xlim(-1, 3)
    plt.ylabel('PESQ値の平均', fontsize=14)
    plt.ylim(0, 5)
    fig.savefig('PESQ'+'('+dtype+')')

    # STOIのグラフ
    fig = plt.figure()
    plt.bar(x_axis, stoi_dict['mean'],
            yerr=stoi_dict['error'], width=0.5, capsize=4, color=colorlist)
    plt.title('STOI'+'('+dtype+')', fontsize=16)
    plt.xlabel('学習に使用したデータ', fontsize=14)
    plt.xlim(-1, 3)
    plt.ylabel('STOIの平均', fontsize=14)
    plt.ylim(0, 1)
    fig.savefig('STOI'+'('+dtype+')')

    # VUV_MATCHのグラフ
    fig = plt.figure()
    plt.bar(x_axis, vuv_match_dict['mean'],
            yerr=vuv_match_dict['error'], width=0.5, capsize=4, color=colorlist)
    plt.title('有声/無声判定の一致率'+'('+dtype+')', fontsize=16)
    plt.xlabel('学習に使用したデータ', fontsize=14)
    plt.xlim(-1, 3)
    plt.ylim(0, 100)
    plt.ylabel('一致率', fontsize=14)
    fig.savefig('有声・無声判定の一致率'+'('+dtype+')')

    # VUV_ERRORのグラフ
    fig = plt.figure()
    plt.bar(x_axis, vuv_error_dict['mean'],
            yerr=vuv_error_dict['error'], width=0.5, capsize=4, color=colorlist)
    plt.title('有声/無声判定のエラー率'+'('+dtype+')', fontsize=16)
    plt.xlabel('学習に使用したデータ', fontsize=14)
    plt.xlim(-1, 3)
    plt.ylim(0, 10)
    plt.ylabel('エラー率', fontsize=14)
    fig.savefig('有声・無声判定のエラー率'+'('+dtype+')')

    # RMSのグラフ
    fig = plt.figure()
    plt.bar(x_axis, rms_dict['mean'], yerr=rms_dict['error'], width=0.5, capsize=4, color=colorlist)
    plt.title('有声部分のRMSE'+'('+dtype+')', fontsize=16)
    plt.xlabel('学習に使用したデータ', fontsize=14)
    plt.xlim(-1, 3)
    plt.ylabel('RMSE', fontsize=14)
    fig.savefig('有声部分のRMSE'+'('+dtype+')')

    # 相関係数のグラフ
    fig = plt.figure()
    plt.bar(x_axis, corr_dict['mean'],
            yerr=corr_dict['error'], width=0.5, capsize=4, color=colorlist)
    plt.title('有声部分の相関係数'+'('+dtype+')', fontsize=16)
    plt.xlabel('学習に使用したデータ', fontsize=14)
    plt.xlim(-1, 3)
    plt.ylabel('相関係数', fontsize=14)
    plt.ylim(0, 1)
    fig.savefig('有声部分の相関係数'+'('+dtype+')')