import numpy as np

import optimal_cut


def test_cumsum():
    x = np.array([1, 2, 3])
    s = np.cumsum(x)
    assert s[2 - 1] == x[:2].sum()


def test_optimal_cut():
    np.random.seed(123)
    n = 10
    gamma = 0.001
    x = np.random.random(size=(n,))
    y = x * 20 + np.random.random(size=(n,))
    values = y[np.argsort(x)]
    cut_i = optimal_cut.find_cut(values, gamma)[0]
    naive_i = optimal_cut.find_cut_naive(values, gamma)[0]
    assert cut_i == naive_i
