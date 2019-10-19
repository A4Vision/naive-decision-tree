"""
t : leaf of the tree
    Minimize
        SUM_t(Lt * Var(Xt) + gamma * Wt ** 2) + gamma * T
"""
import numpy as np


def calc_score(values, gamma):
    w_t = np.average(values)
    return np.sum((values - w_t) ** 2) + gamma * (w_t ** 2 + 1)


def sort_by(x, values):
    return np.array(x)[np.argsort(values)]


def find_cut_naive_given_discrete(values, discrete, gamma):
    sorted_values = sort_by(values, discrete)
    scores = scores_naive(sorted_values, gamma)[1:]
    scores[np.diff(sorted(discrete)) == 0] = np.inf
    i = np.argmin(scores)
    return np.sort(discrete)[i], scores[i]


def scores_naive(values, gamma):
    res = np.zeros(shape=values.shape[0])
    res[0] = calc_score(values, gamma)
    for i in range(1, values.shape[0]):
        s1 = calc_score(values[:i], gamma)
        s2 = calc_score(values[i:], gamma)
        res[i] = s1 + s2
    return res


def find_cut_naive(values, gamma):
    scores = scores_naive(values, gamma)[1:]
    i = np.argmin(scores)
    return i + 1, scores[i]


def find_cut(values, gamma):
    cs_vals = np.cumsum(values)
    cs_vals_sq = cs_vals ** 2
    cs_sq_values = np.cumsum(values ** 2)
    lengths = np.arange(1, values.shape[0])

    r_cs_sq_values = cs_sq_values[-1] - cs_sq_values
    r_cs_vals = cs_vals[-1] - cs_vals
    r_cs_vals_sq = r_cs_vals ** 2

    r_lengths = values.shape[0] - lengths
    assert lengths[0] == 1. and r_lengths[-1] == 1
    assert r_cs_sq_values[-1] == 0 == r_cs_vals_sq[-1]
    assert r_lengths[0] == values.shape[0] - 1

    scores_left = cs_sq_values[:-1] + (gamma / (lengths ** 2) - 1 / lengths) * cs_vals_sq[:-1]
    scores_right = r_cs_sq_values[:-1] + (gamma / (r_lengths ** 2) - 1 / r_lengths) * r_cs_vals_sq[:-1]

    scores = scores_left + scores_right + 2 * gamma
    i = np.argmin(scores)
    return i + 1, scores[i]
