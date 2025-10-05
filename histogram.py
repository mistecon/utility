import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import subprocess, textwrap

def func_hist(data, xmin=None, xmax=None, binno=20, density_flg=True):
    """
    等幅ヒストグラム（集計区間は [xmin, xmax)）
    density_flg=True: 確率密度（PDF）を返す。面積=1 になるように height = counts / (N*width)。
    戻り値:
        densities_or_counts (binno,), widths (binno,), centers (binno,)
    """
    if xmin is None:
        xmin = data.min()
    if xmax is None:
        xmax = data.max()

    bins = np.linspace(xmin, xmax, binno+1)
    #データをxmin, xmaxに含まれる範囲に絞る
    data_clip = data[(data>=xmin)&(data<xmax)]
    data_dg = np.digitize(data_clip, bins, right=True) #right=Trueはビンの左端を含み、右端を含まない
    counts = pd.Series(data_dg).value_counts().reindex(index=np.arange(1, binno+1)).fillna(0).to_numpy()
    widths = bins[1:] - bins[:-1]
    if counts.sum()>0:
        densities = counts/(widths*counts.sum())
    else:
        densities = np.zeros(binno)
    #print((densities*widths).sum())
    centers = (bins[1:] + bins[:-1])/2
    if density_flg == True:
        return densities, widths, centers
    else:
        return counts, widths, centers

def func_hist_log(data, xmin=None, xmax=None, binno=20, density_flg=True):
    """
    data : numpy array (size n)
    """
    if xmin is None:
        xmin = data.min()
    if xmax is None:
        xmax = data.max()
    if xmin<=0:
        print("xmin should be positive")
        return None, None, None

    logbins = np.linspace(np.log(xmin), np.log(xmax), binno+1)

    #データをxmin, xmaxに含まれる範囲に絞る
    data_clip = data[(data>=xmin)&(data<xmax)]
    #データを何番目のbinかの数字に変換
    data_dg = np.digitize(data_clip, np.exp(logbins), right=True) #right=Trueはビンの左端を含み、右端を含まない
    counts = pd.Series(data_dg).value_counts().reindex(index=np.arange(1, binno+1)).fillna(0).to_numpy()
    widths = np.exp(logbins[1:]) - np.exp(logbins[:-1])
    if counts.sum()>0:
        densities = counts/(widths*counts.sum())
    else:
        densities = np.zeros(binno)
    print((densities*widths).sum())
    centers = (np.exp(logbins[1:]) + np.exp(logbins[:-1]))/2
    if density_flg == True:
        return densities, widths, centers
    else:
        return counts, widths, centers

