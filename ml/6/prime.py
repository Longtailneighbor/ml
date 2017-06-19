#!/usr/bin/python
# -*- coding:utf-8 -*-

import math
from time import time


def is_prime(x):
    return 0 not in [x % i for i in range(2, int(math.sqrt(x)) + 1)]


def is_prime2(x):
    if x <= 1:
        return False
    elif x < 4:
        return True
    else:
        r = int(math.sqrt(x))
        for i in range(2, r + 1):
            if x % i == 0:
                return False


# 利用filter 和 lambda
is_prime4 = (lambda x: 0 not in [x % i for i in range(2, int(math.sqrt(x)) + 1)])


def compare():
    a = 2
    b = 100000

    a = 2
    b = 100000

    # 方法1： 直接计算
    t_begin = time()
    print [p for p in range(a, b) if 0 not in [p % d for d in range(2, int(math.sqrt(b)) + 1)]]
    t_end = time()
    print t_end - t_begin

    # 方法2： 利用filter
    t_begin = time()
    print filter(is_prime, range(a, b))
    t_end = time()
    print t_end - t_begin

    # 方法3： 利用filter和lambda
    t_begin = time()
    print filter(is_prime4, range(a, b))
    t_end = time()
    print t_end - t_begin

    # 方法4：定义和filter
    p_list = []
    t_begin = time()
    p_list = filter(is_prime, range(2, b))
    t_end = time()
    print t_end - t_begin
    print p_list
