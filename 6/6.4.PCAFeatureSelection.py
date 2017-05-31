#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split


def extend(a, b):
    return 1.05 * a - 0.05 * b, 1.05 * b - 0.05 * a


if __name__ == '__main__':
    pca = True

    columns = np.array([u'sepal_length', u'sepal_width', u'petal_length', u'petal_width', u'type'])

    data = pd.read_csv('iris.data', header=None)
    data.rename(columns=dict(zip(np.arange(5), columns)), inplace=True)
    data[u'type'] = pd.Categorical(data[u'type']).codes

    x = data[columns[:-1]]
    y = data[columns[-1]]

    if pca:
        pca = PCA(n_components=2, whiten=True, random_state=0)
        x = pca.fit_transform(x)

        print 'Variance: ', pca.explained_variance_
        print 'Variance Ratio: ', pca.explained_variance_ratio_

        x1_label, x2_label = u'component1', u'compontent2'
        title = u'Iris Data - PCA Dimensionality Reduction'
    else:
        fs = SelectKBest(chi2, k=2)
        fs.fit(x, y)

        idx = fs.get_support(indices=True)
        print 'fs.get_support() = ', idx

        x = x[idx]
        x = x.values  # Convert DataFrame to ndarray

        x1_label, x2_label = columns[idx]
        title = u'Iris Data - Feature Selection'

    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g', 'b'])

    # Show the grid
    plt.scatter(x[:, 0], x[:, 1], s=30, c=y, marker='o', cmap=cm_dark)
    plt.xlabel(x1_label)
    plt.ylabel(x2_label)
    plt.title(title)
    plt.grid(b=True, ls=':')
    plt.show()

    x, x_test, y, y_test = train_test_split(x, y, train_size=0.7)
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=True)),
        ('Ir', LogisticRegressionCV(Cs=np.logspace(-3, 4, 8), cv=5, fit_intercept=False))
    ])
    model.fit(x, y)
    y_hat = model.predict(x)
    y_test_hat = model.predict(x_test)

    print 'Optimal parameter: ', model.get_params('Ir')['Ir'].C_
    print 'Tain Data Accuracy: ', metrics.accuracy_score(y, y_hat)
    print 'Test Data Accuracy: ', metrics.accuracy_score(y_test, y_test_hat)

    # 横纵个采样多少值
    N, M = 500, 500
    x1_min, x1_max = extend(x[:, 0].min(), x[:, 0].max())  # x[:, 0]为获取第一列
    x2_min, x2_max = extend(x[:, 1].min(), x[:, 1].max())
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    y_hat = model.predict(x_show)    # predict
    y_hat = y_hat.reshape(x1.shape)  # 使与之输入的形状相同

    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], s=30, c=y, edgecolors='k', cmap=cm_dark)  # 样本的显示
    plt.xlabel(x1_label)
    plt.ylabel(x2_label)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    patchs = [mpatches.Patch(color='#77E0A0', label='Iris-setosa'),
              mpatches.Patch(color='#FF8080', label='Iris-versicolor'),
              mpatches.Patch(color='#A0A0FF', label='Iris-virginica')]
    plt.legend(handles=patchs, fancybox=True, framealpha=0.8, loc='lower right')
    plt.title(u'Iris Logistic Regression', fontsize=17)
    plt.grid(b=True, ls=':')
    plt.show()
