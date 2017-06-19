#!/usr/bin/python
# -*- encoding: utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor



if __name__ == '__main__':

    data = pd.read_csv('C0904.csv', header=0)
    x = data['H2O'].values

    # Find the abnormal data
    width = 500
    delta = 10
    eps = 0.15
    N = len(x)
    p = []
    abnormal = []
    for i in np.arange(0, N - width, delta):
        s = x[i:i + width]
        p.append(np.ptp(s))
        if np.ptp(s) > eps:
            abnormal.append(range(i, i + width))
    abnormal = np.array(abnormal).flatten()
    abnormal = np.unique(abnormal)

    plt.subplot(131)
    plt.plot(x, 'r-', label=u'The Origin Data')
    plt.title(u'The Real Emissions')
    plt.legend(loc='upper right')
    plt.grid(b=True)

    plt.subplot(132)
    t = np.arange(N)
    plt.plot(abnormal, x[abnormal], 'go', markeredgecolor='g', ms=3, label=u'abnormal')
    plt.title(u'Abnormal Detect')
    plt.legend(loc='upper right')
    plt.grid(b=True)

    # predict
    plt.subplot(133)
    select = np.ones(N, dtype=np.bool)
    select[abnormal] = False
    t = np.arange(N)
    dtr = DecisionTreeRegressor(criterion='mse', max_depth=10)
    br = BaggingRegressor(dtr, n_estimators=10, max_samples=0.3)
    br.fit(t[select].reshape(-1, 1), x[select])
    y = br.predict(np.arange(N).reshape(-1, 1))
    y[select] = x[select]

    plt.plot(x, 'g--', lw=1, label=u'The Origin Data')
    plt.plot(y, 'r-', lw=1, label=u'The Correction Data')
    plt.legend(loc='upper right')
    plt.title(u'Correct the abnormal Data', fontsize=18)
    plt.grid(b=True)

    plt.tight_layout(1.5, rect=(0,0,1,0.95))
    plt.show()
