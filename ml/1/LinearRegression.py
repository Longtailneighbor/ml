# -- coding: utf-8 --
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 收集、准备数据
data = pd.read_csv('Advertising.csv')

# 2. 使用pandas来构建X（特征向量）和Y（标签列）
# Create a python list of feature name
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]

# print the first 5 rows
print X.head()

# check the type and shape of X
print type(X)
print X.shape

# Select a series from the DataFrame
Y = data['Sales']
print Y.head()

# 3. 构建训练集与测试集
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
# 默认分割为75%的训练集，25%的测试集
print X_train.shape
print X_test.shape
print Y_train.shape
print Y_test.shape

# 4. sklearn 的线性回归
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
model = linreg.fit(X_train, Y_train)
print model
print linreg.intercept_
print linreg.coef_
# pair the feature names with the coefficients
zip(feature_cols,linreg.coef_)

# 5. 预测
y_pred = linreg.predict(X_test);
print y_pred
print type(y_pred)

# 6. 回归问题的评价测度

print type(y_pred), type(Y_test)
print len(y_pred), len(Y_test)
print y_pred.shape, Y_test.shape

sum_mean = 0;
for i in range(len(y_pred)):
    sum_mean += (y_pred[i] - Y_test.values[i])**2

print "RMSE by hand: ", np.sqrt(sum_mean / len(y_pred))

# 7. 作图
plt.figure()
plt.plot(range(len(y_pred)), y_pred, 'b', lable="predict")
plt.plot(range(len(Y_test)), Y_test, 'r', lable="test")
plt.legend(loc="upper right")
plt.xlabel("the number of sales")
plt.ylabel("value of sales")
plt.show()