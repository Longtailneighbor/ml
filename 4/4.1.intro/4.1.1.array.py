#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import time
import math

# 导入Num函数库，一般都是这样的形式（包括别名np，几乎是约定俗称的）

# 标准Python的列表list中，元素本质是对象
# 如：L = [1, 2, 3]，需要三个指针和三个整数对象，对于数值运算比较浪费内存和CPU
# 因此Numpy提供了ndarray(N - Dimensional array object)对象，存储单一数据类型的多维数组

if __name__ == "__main__":

    ## 通过array函数传递list对象
    L = [1, 2, 3, 4, 5, 6]
    arr = np.array(L)
    print "Origin List: ", L
    print "Type: ", type(L)
    print "Array: ", arr
    print "Type: ", type(arr)
    # 若传递的是多层嵌套list，将转换为多维数组
    multi_arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print  multi_arr

    # 通过shape获取数组大小
    print arr.shape
    print multi_arr.shape

    # 通过shape修改数组大小
    # 注：从(3,4)改为(4,3)并不是对数组进行转置，而只是改变每个轴的大小，数组元素在内存中的位置并没有改变
    multi_arr.shape = 4, 3
    print multi_arr

    # 当某个轴为-1时，将根据数组元素的个数自动计算此轴的长度。-1可以看作是缺省值
    multi_arr.shape = 2, -1
    print multi_arr
    print multi_arr.shape

    # reshape, 创建新的数组，元数组shape保持不变
    multi_arr_copy = multi_arr.reshape(4, -1)
    print multi_arr
    print multi_arr_copy

    # 数组的元素类型可以通过dtype属性获得
    print arr.dtype
    print multi_arr.dtype

    # 创建数组时，通过dtype指定数组数据类型
    arr_float = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float)
    arr_complext = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.complex)

    # 如果更改元素类型，可以使用astype安全的转换
    arr_convert_int = arr_float.astype(np.int)

    ## 使用函数创建数组
    # 如果生成一定规则的数据，可以使用Numpy提供的专门arange函数，类似python的range(start, end, step)
    # 和python的range类似，arange不包括终止值。但arange可以生成浮点类型，而range只能是整数类型
    arr_arange = np.arange(1, 10, 10)

    # linspace函数，等差函数，指定起始值、终止值和元素个数来创建数组，缺省包括终止值
    arr_linspace = np.linspace(1, 10, 10)
    arr_linspace_endpoint_false = np.linspace(1, 10, 10, endpoint=False)

    # logspace, 等比函数
    # 下面函数创建起始值为10^1，终止值为10^2，有10个数的等比数列
    arr_logspace = np.logspace(1, 2, 9, endpoint=True)

    # base为2
    arr_logspace_base = np.logspace(1, 2, 9, endpoint=True, base=2)

    # fromstring, fromfile, frombuffer
    str = 'abcde'
    arr_s = np.fromstring(str, dtype=np.int8)

    ## 索引数组
    arr = np.arange(10)
    # 获取某个元素
    print arr[3]
    # 切片[3, 6)左闭右开
    print arr[3:6]
    # 省略开始下标表示从0开始，省略结束下标表示直到数组结束
    print arr[:5]
    print arr[3:]
    # 下标为负表示从后向前数
    print arr[3:-1]
    # 步长为2
    print arr[3:6:2]
    # 步长为-1，即翻转
    print arr[::-1]
    # 切片数据是原数组的一个视图，与原数组共享内容空间，可以直接修改元素值


    ## 整数数组索引
    # 整数数组索引：当使用整数序列对数组元素进行索引时，将使用整数序列中的每个元素作为下标，整数序列可以是list或者ndarray
    # 使用整数序列作为下标索引的数组不合原始数组共享数据空间
    arr_logspace = np.logspace(0, 9, 10, base=2)
    i = np.arange(0, 10, 2)
    arr_index = arr_logspace[i]

    ## 布尔数组索引
    # 使用布尔数组i作为下标存取数组a中的元素：返回数组a中所有在数组b中对应下标为True的元素
    # 生成10个满足[0,1)中均匀分布的随机数
    arr_bools = np.random.rand(10)
    print arr_bools
    # 大于0.5的元素索引
    print arr_bools > 0.5
    # 大于0.5的元素
    print arr_bools[arr_bools > 0.5]
    # 将原数组中大于0.5的元素赋值为0.5
    arr_bools[arr_bools > 0.5] = 0.5

    ## 二维数组的切片
    arr_x = np.arange(0, 60, 10)                  # 行向量
    arr_y = arr_x.reshape(-1, 1)                  # 列向量
    arr_x_increase = np.arange(6)                 # [0, 1, 2, 3, 4, 5]
    arr_two_dimensional = arr_y + arr_x_increase
    # [[ 0  1  2  3  4  5]
    #  [10 11 12 13 14 15]
    #  [20 21 22 23 24 25]
    #  [30 31 32 33 34 35]
    #  [40 41 42 43 44 45]
    #  [50 51 52 53 54 55]]

    # 合并
    a = np.arange(0, 60, 10).reshape(-1, 1) + np.arange(6)

    ## 二维数组的切片
    print a[[0, 1, 2], [2, 3, 4]]
    print a[4, [2, 3, 4]]
    print a[4:, [2, 3, 4]]
    i = np.array([True, False, True, False, False, True])
    print a[i]
    print a[i, 3]

    # 获取第一列数据
    print a[:, 0]

    ## numpy与python math库的对比
    for i in np.logspace(0, 7, 10):
        i = int(i)
        x = np.linspace(0, 10, i)

        start = time.clock()
        y = np.sin(x)
        end = time.clock()
        t1 = end - start

        x = x.tolist()
        start = time.clock()
        for i, j in enumerate(x):
            x[i] = math.sin(j)
        end = time.clock()
        t2 = end - start

        print i, ": ", t1, t2, t2 / t1

    ## 元素去重
    arr = np.array(1, 2, 3, 4, 5, 5, 7, 3, 2, 2, 8, 8)
    np.unique(arr)
    # 二维数组的去重
    # 方案1：转换为虚数
    arr_multi = np.array(((1, 2), (3, 4), (5, 6), (1, 3), (3, 4), (7, 6)))
    r, i = np.split(arr_multi, (1, ), axis=1)
    x = r + i * 1j
    x = arr_multi[:, 0] + arr_multi[:, 1] * 1j
    print '转换成虚数：', x
    print '虚数去重后：', np.unique(x)
    print np.unique(x, return_index=True)   # 思考return_index的意义
    idx = np.unique(x, return_index=True)[1]
    # 方案2：利用set
    np.array(list(set([tuple(t) for t in arr_multi])))

    ## stack and axis
    a = np.arange(1, 10).reshape((3, 3))
    b = np.arange(11, 20).reshape((3, 3))
    c = np.arange(101, 110).reshape((3, 3))
    print 'a = \n', a
    print 'b = \n', b
    print 'c = \n', c
    print 'axis = 0 \n', np.stack((a, b, c), axis=0)
    print 'axis = 1 \n', np.stack((a, b, c), axis=1)
    print 'axis = 2 \n', np.stack((a, b, c), axis=2)

    a = np.arange(1, 10).reshape(3,3)
    print a
    b = a + 10
    print b
    print np.dot(a, b)
    print a * b

    a = np.arange(1, 10)
    print a
    b = np.arange(20,25)
    print b
    print np.concatenate((a, b))
