#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np

def calc_convariance(x, y):
    if len(x) != len(y):
        return

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    sum = 0
    for i in range(0, len(x)):
        sum += ((x[i] - x_mean) * (y[i] - y_mean))

    sigma = sum / (len(x) - 1)

    return sigma

if __name__ == "__main__":

    N = 2
    tip = u'一次函数关系'
    x = np.random.rand(N)
    y = np.zeros(N) + np.random.rand(N)*0.001

    print np.cov(x, y)[0,1]
    print calc_convariance(x, y)