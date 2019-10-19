import time
from typing import List

import numpy as np
import pandas as pd
import xgboost
import re

from tree.naive_train.optimal_cut import find_cut_naive_given_discrete
from tree.optimized_train import optimized_train_tree, utils
from tree.optimized_train.decision_rule_selection import get_top_by_scores
from tree.optimized_train.scores_calculator import calculate_features_scores
from tree.optimized_train.value_to_bins import ValuesToBins


def test_take_top_k():
    top = get_top_by_scores(np.array([10, 2, 3, 4, 5]), np.array([0.1, -1, 2, 3, 4]), 2)
    assert set(top) == {10, 2}


def test_calculate_features_scores():
    y = np.random.random(10)
    range_size = 6
    np.random.seed(123)
    x = np.random.randint(13, 13 + range_size, size=(10, 30))
    for n_bins in (3, 4, 7, 11, 15):
        v = ValuesToBins(x, n_bins)
        bins = v.get_bins(x)
        assert bins.shape == x.shape
        scores = calculate_features_scores(bins, y)
        naive_scores = np.array([find_cut_naive_given_discrete(y, bins[:, i], 0)[1] for i in range(x.shape[1])])
        assert (scores > naive_scores - 0.00001).all()
        error = (scores - naive_scores) / (naive_scores + scores)
        assert np.max(error) < 0.0001


def _test_train_tree_optimized(x, y, **params_args):
    params = {'max_depth': 5, 'gamma': 0.0001, 'feature_pruning_method': 'dynamic'}
    # rs_log2 = np.array([8, 4., 0])
    rs_log2 = np.array([8, 4., 0])
    ks_log2 = np.array([0., 2, 4])
    params.update({'lines_sample_ratios': 2 ** -rs_log2,
                   'features_sample_ratios': 2 ** -ks_log2})
    params.update(params_args)
    tree = optimized_train_tree.train(x, y, params)
    print(tree.root())
    prediction = tree.predict_many(x)
    diff = prediction - y
    clean_diff = _drop_outliers(diff, 0.005)
    print('average squared residue', np.average(clean_diff ** 2))
    # assert np.abs(clean_diff).max() < 0.1


def test_train_tree_optimized_level1():
    x = np.random.normal(size=(50000, 128))
    y = (x.T[7] > 0.1) * 5 + (x.T[2] < 0.01) * 3 + np.random.random(size=(x.shape[0])) * 0.01
    _test_train_tree_optimized(x, y)


def test_train_tree_optimized_level2():
    x = np.random.normal(size=(50000, 128))
    y = (x.T[3] > 0.2) * 3 + ((x.T[2] < 0.01) | (x.T[1] > 0.1)) * 3 + np.random.random(size=(x.shape[0])) * 0.01
    _test_train_tree_optimized(x, y)



def test_train_tree_optimized_level3():
    x = np.random.randint(0, 10, size=(20000, 128)) * 0.1
    y = (x.T[10] > 0.3) * 1 + ((x.T[100] < 0.01) & (x.T[1] + x.T[2] < 0.5)) * 2 + np.random.random(
        size=(x.shape[0])) * 0.01
    t = time.time()
    _test_train_tree_optimized(x, y, max_depth=4)
    my_runtime = time.time() - t
    t = time.time()
    regressor = xgboost.XGBRegressor(gamma=0., max_depth=5, learning_rate=1., base_score=0.5,
                                     n_estimators=1, reg_lambda=0.,
                                     min_child_weight=0,
                                     tree_method='exact',
                                     **{'lambda': 0})
    regressor.fit(x, y)
    xgboost_runtime = time.time() - t

    diff = y - regressor.predict(x)
    clean_diff = _drop_outliers(diff, 0.005)
    print(utils.booster_text(regressor.get_booster(), 0.5))
    print('my runtime', my_runtime)
    print('xgboost runtime', xgboost_runtime)
    print('average squared residue by xgboost', np.average(clean_diff ** 2))


def show_xgboost_does_not_implement_fast_pruning():
    times = []
    ns = range(20000, 100000, 10000)
    for n in ns:
        x = np.random.randint(0, 10, size=(n, 128)) * 0.1
        y = (x.T[10] > 0.3) * 1 + ((x.T[100] < 0.01) & (x.T[1] + x.T[2] < 0.5)) * 2 + np.random.random(
            size=(x.shape[0])) * 0.01
        t = time.time()
        regressor = xgboost.XGBRegressor(max_depth=2, learning_rate=1., base_score=0.5,
                                         n_estimators=1, reg_lambda=0., min_child_weight=0)
        regressor.fit(x, y)
        xgboost_runtime = time.time() - t
        times.append(xgboost_runtime)
    print(times)
    print(list(ns))


def _drop_outliers(values, percent):
    assert 0 < percent < 1
    s = pd.Series(values)
    return s[(s >= s.quantile(percent)) &
             (s <= s.quantile(1 - percent))]

# Problems occur:
#    * Calculation in deep nodes is in-stable:
#    with few samples, the whole method does not work.
#    * Calculation in deep nodes is very expensive, because we cannot differentiate the optimal leaf easily by pruning.
#
# Problems with comparing to xgboost:
#  * xgboost does not always split - need to see why, and how to implement
# a similar mechanism.
#  *  We get non-comparable performance to xgboost due to binning.
