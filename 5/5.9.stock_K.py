#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    np.set_printoptions(suppress=True, linewidth=100, edgeitems=5)

    data = np.loadtxt('SH600000.txt', dtype=np.float, delimiter='\t', skiprows=2, usecols=(1, 2, 3, 4))
    data = data[:50]

    N = len(data)

    nums = np.arange(1, N+1).reshape((-1, 1))
    data = np.hstack((nums, data))

    fig, ax = plt.subplots(facecolor='w')
    candlestick_ohlc(ax, data, 0.6)
    plt.grid()
    plt.title(u'股票K线图', fontsize=18)
    plt.show()
