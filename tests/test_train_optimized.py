import time

import numpy as np
import pandas as pd
import pytest
import xgboost
from sklearn.model_selection import train_test_split

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


def _test_train_tree_optimized(x, y, validate=True, **params_args):
    params = {'max_depth': 5, 'gamma': 0.0001, 'feature_pruning_method': 'dynamic'}
    rs_log2 = np.array([8, 4., 0])
    ks_log2 = np.array([0., 2, 4])
    params.update({'lines_sample_ratios': 2 ** -rs_log2,
                   'features_sample_ratios': 2 ** -ks_log2})
    params.update(params_args)
    tree, selector = optimized_train_tree.train(x, y, params)
    prediction = tree.predict_many(x)
    diff = prediction - y
    clean_diff = _drop_outliers(diff, 0.005)
    if validate:
        assert np.abs(clean_diff).max() < 0.1
    return tree, selector


def test_train_tree_optimized_level1():
    x = np.random.normal(size=(50000, 128))
    y = (x.T[7] > 0.1) * 5 + (x.T[2] < 0.01) * 3 + np.random.random(size=(x.shape[0])) * 0.01
    _test_train_tree_optimized(x, y)


def test_train_tree_optimized_level2():
    x = np.random.normal(size=(50000, 128))
    y = (x.T[3] > 0.2) * 3 + ((x.T[2] < 0.01) | (x.T[1] > 0.1)) * 3 + np.random.random(size=(x.shape[0])) * 0.01
    _test_train_tree_optimized(x, y)


def get_average_diff(y, prediction):
    return np.average((y - prediction) ** 2)


def test_train_tree_optimized_level3():
    np.random.seed(123)
    x = np.random.randint(0, 10, size=(20000, 128)) * 0.1
    y = (x.T[10] > 0.3) * 1 + ((x.T[100] < 0.01) & (x.T[1] + x.T[2] < 0.5)) * 2 + np.random.random(
        size=(x.shape[0])) * 0.01

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    depth = 3
    _compare_xgboost_with_my_implementation(x_train, x_test, y_train, y_test, depth)


def _compare_xgboost_with_my_implementation(x_train, x_test, y_train, y_test,
                                            depth):
    t = time.time()
    my_regressor, selector = _test_train_tree_optimized(x_train, y_train, max_depth=depth, validate=False)
    my_runtime = time.time() - t
    my_diff = get_average_diff(y_test, my_regressor.predict_many(x_test))

    t = time.time()
    base_score = np.average(y_train)
    xgboost_regressor = xgboost.XGBRegressor(gamma=0., max_depth=depth,
                                             learning_rate=1., base_score=base_score,
                                             n_estimators=1, reg_lambda=0.,
                                             min_child_weight=0,
                                             tree_method='exact',
                                             **{'lambda': 0})
    xgboost_regressor.fit(x_train, y_train)
    xgboost_runtime = time.time() - t
    xgboost_diff = get_average_diff(y_test, xgboost_regressor.predict(x_test))
    print(my_regressor.root())
    print(utils.booster_text(xgboost_regressor.get_booster(), base_score))
    print('xgboost runtime', xgboost_runtime)
    print('my      runtime', my_runtime)
    print('xgboost avg squared residue', xgboost_diff)
    print('my      avg squared residue', my_diff)
    assert pytest.approx(xgboost_diff, rel=0.05) == my_diff
    return selector

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
