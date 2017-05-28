#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import prime

if __name__ == "__main__":
    a = 1600
    b = 1700

    p = np.array(filter(prime.is_prime, range(2, b)))
    p = p[p >= a]

    p_rate = float(len(p)) / float(b - a + 1)

    print "The prime number probability: ", p_rate, '\t',
    print "The fair Odds: ", 1/p_rate
    print "The composite number probability: ", 1 - p_rate, '\t',
    print "The fair Odds: ", 1 / (1 - p_rate)
