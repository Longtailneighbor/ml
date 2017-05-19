import numpy as np


# calculate the machine epsilon
def mEps(func = float):
    m_eps = func(1)
    while func(1) + func(m_eps) != func(1):
        m_eps_last = m_eps
        m_eps = func(m_eps) / func(2)
    return m_eps_last

print np.finfo(float).eps
print np.finfo(np.float32).eps
print
print mEps(float)
print mEps(np.float32)