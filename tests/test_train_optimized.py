import numpy as np
import pandas as pd

from tree.naive_train.optimal_cut import find_cut_naive_given_discrete
from tree.optimized_train import optimized_train_tree
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


def test_train_tree_optimized():
    np.random.seed(123)
    x = np.random.normal(size=(50000, 128))
    y = (x.T[7] > 0.1) * 5 + (x.T[2] < 0.01) * 3 + np.random.random(size=(x.shape[0])) * 0.01
    params = {'max_depth': 2, 'gamma': 0.1, 'feature_pruning_method': 'dynamic'}
    rs_log2 = np.array([8, 4., 0])
    ks_log2 = np.array([0., 2, 4])
    params.update({'lines_sample_ratios': 2 ** -rs_log2,
                   'features_sample_ratios': 2 ** -ks_log2})
    tree = optimized_train_tree.train(x, y, params)
    print(tree.root())
    prediction = tree.predict_many(x)
    diff = prediction - y
    clean_diff = _drop_outliers(diff, 0.005)
    assert np.abs(clean_diff).max() < 0.1


def _drop_outliers(values, percent):
    assert 0 < percent < 1
    s = pd.Series(values)
    return s[(s >= s.quantile(percent)) &
             (s <= s.quantile(1 - percent))]
