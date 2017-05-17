#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = 'SimHei'

if __name__ == "__main__":
    x1, x2 = np.mgrid[-5:5:51j, -5:5:51j]
    x = np.stack((x1, x2), axis=2)

    fig = plt.figure(figsize=(9, 8), facecolor='w')

    sigma = (np.identity(2), np.diag((3, 3)), np.diag((2, 5)), np.array(((2, 1), (2, 5))))
    for i in np.arange(4):
        norm = stats.multivariate_normal((0, 0), sigma[i])
        y = norm.pdf(x)
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.plot_surface(x1, x2, y, cmap=cm.Accent, rstride=2, cstride=2, alpha=0.5, lw=0.3)

    plt.suptitle(u'二元高斯分布方差比较', fontsize=18)
    plt.tight_layout(1.5)
    plt.show()
