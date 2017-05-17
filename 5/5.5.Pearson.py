#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import division

import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = 'SimHei'


def rotate(x, y, theta=45):
    data = np.vstack((x, y))
    mu = np.mean(data, axis=1)
    mu.reshape((-1, 1))
    data -= mu

    theta *= (np.pi / 180)
    c = np.cos(theta)
    s = np.sin(theta)
    m = np.array(((c, -s), (s, c)))
    return m.dot(data) + mu


# 皮尔逊相关系数定义为两个变量之间的协方差和标准差的商
# 标准差 = 方差的平方根
# 方差 = 平方的平均 - 平均的平方
def calc_pearson(x, y):
    xstd = np.sqrt(np.mean(x ** 2) - np.mean(x) ** 2)
    ystd = np.sqrt(np.mean(y ** 2) - np.mean(y) ** 2)
    cov = np.cov(x, y, bias=True)[0,1]
    return cov / (xstd * ystd)


def pearson(x, y, tip):
    clrs = list('rgbmycrgbmycrgbmycrgbmyc')
    plt.figure(figsize=(10, 8), facecolor='w')

    for i, theta in enumerate(np.linspace(0, 90, 6)):
        x_rotate, y_rotate = rotate(x, y, theta)
        _pearson = stats.pearsonr(x, y)[0]
        print '旋转角度：', theta, 'Pearson相关系数：', _pearson
        str = u'相关系数：%.3f' % _pearson
        plt.scatter(x_rotate, y_rotate, s=40, alpha=0.9, linewidths=0.5, c=clrs[i], marker='o', label=str)

    plt.legend(loc='upper left', shadow=True)
    plt.xlabel(u'X')
    plt.ylabel(u'Y')
    plt.title(u'Pearson相关系数与数据分布：%s' % tip, fontsize=18)
    plt.grid(b=True)
    plt.show()


if __name__ == "__main__":

    N = 2
    tip = u'一次函数关系'
    x = np.random.rand(N)
    y = np.zeros(N) + np.random.rand(N)*0.001

    # tip = u'二次函数关系'
    # x = np.random.rand(N)
    # y = x ** 2

    # tip = u'正切关系'
    # x = np.random.rand(N) * 1.4
    # y = np.tan(x)

    # tip = u'二次函数关系'
    # x = np.linspace(-1, 1, 101)
    # y = x ** 2

    # tip = u'椭圆'
    # x, y = np.random.rand(2, N) * 60 - 30
    # y /= 5
    # idx = (x**2 / 900 + y**2 / 36 < 1)
    # x = x[idx]
    # y = y[idx]

    pearson(x, y, tip)

