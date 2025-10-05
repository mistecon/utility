!apt-get -y update
!apt-get -y install gnuplot
!pip -q install PyGnuplot gnuplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import subprocess, textwrap

def gnuhist(counts_densities, widths, centers, outfpath:Path, logxy_flg=False):
    outfpath = str(outfpath).replace('\\','/')
    logscale = {True:"set logscale xy", False:""}[logxy_flg]
    data_block = "\n".join(f"{x} {y} {w}" for x,y,w in zip(centers, counts_densities, widths))
    print(outfpath, logscale)
    xmin = {True:centers[0]/2, False:(centers[0]-widths[0])/1.1}[logxy_flg]
    xmax = {True:centers[-1]*2, False:(centers[-1]+widths[-1])*1.1}[logxy_flg]

    ymin = {True:counts_densities[counts_densities!=0].min()/2, False:0}[logxy_flg]
    ymax = {True:counts_densities.max()*2, False:counts_densities.max()*1.1}[logxy_flg]

    script = textwrap.dedent(f"""
        set term pngcairo size 800,500
        set output '{outfpath}'
        {logscale}
        set xrange [{xmin}:{xmax}]
        set yrange [{ymin}:{ymax}]
        set style fill solid 0.5
        set boxwidth 0.9 relative
        plot '-' using 1:2:(0.9*$3) with boxes notitle
        {data_block}
        e
    """)
    subprocess.run(["gnuplot"], input=script.encode("utf-8"), check=True)

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

