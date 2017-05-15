#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import division

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import cm


# 手动计算
def calc_statistics(x):
    n = x.shape[0]  # 样本个数
    m, m2, m3, m4 = 0, 0, 0, 0
    for t in x:
        m += t
        m2 += t * t
        m3 += t ** 3
        m4 += t ** 4
    m /= n
    m2 /= n
    m3 /= n
    m4 /= n

    mu = m
    sigma = np.sqrt(m2 - mu * mu)
    skew = (m3 - 3 * mu * m2 + 2 * mu ** 3) / sigma ** 3
    kurtosis = (m4 - 4 * mu * m3 + 6 * mu * mu * m2 - 4 * mu ** 3 * mu + mu ** 4) / sigma ** 4 - 3
    return mu, sigma, skew, kurtosis


# 使用系统函数
def calc_statistics_lib(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    skew = stats.skew(x)
    kurtosis = stats.kurtosis(x)
    return mu, sigma, skew, kurtosis


if __name__ == '__main__':
    d = np.random.randn(100000)

    mu, sigma, skew, turtosis = calc_statistics(d)
    print "Calcutate           : mean = %.5f, standard Deviation = %.5f, skew = %.5f, turtosis = %.5f" % (mu, sigma, skew, turtosis)

    mu, sigma, skew, turtosis = calc_statistics_lib(d)
    print "Calcutate by library: mean = %.5f, standard Deviation = %.5f, skew = %.5f, turtosis = %.5f"% (mu, sigma, skew, turtosis)

    # 一维直方图
    mpl.rcParams[u'font.sans-serif'] = 'SimHei'
    mpl.rcParams[u'axes.unicode_minus'] = False
    y1, x1, dummy = plt.hist(d, bins=50, normed=True, color='g', alpha=0.75)
    t = np.arange(x1.min(), x1.max(), 0.05)
    y = np.exp(-t ** 2 / 2) / math.sqrt(2 * math.pi)
    plt.plot(t, y, 'r-', lw=2)
    plt.title(u'高斯分布，样本个数：%d' % d.shape[0])
    plt.grid(True)
    plt.show()

    # 二维图像
    N = 30
    density, edges = np.histogramdd(d, bins=[N, N])
    print '样本总数：', np.sum(density)
    density /= density.max()
    x = y = np.arange(N)
    t = np.meshgrid(x, y)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(t[0], t[1], density, c='r', s=15 * density, marker='o', depthshade=True)
    ax.plot_surface(t[0], t[1], density, cmap=cm.Accent, rstride=2, cstride=2, alpha=0.9, lw=0.75)
    ax.set_xlabel(u'X')
    ax.set_ylabel(u'Y')
    ax.set_zlabel(u'Z')
    plt.title(u'二元高斯分布，样本个数：%d' % d.shape[0], fontsize=20)
    plt.tight_layout(0.1)
    plt.show()
