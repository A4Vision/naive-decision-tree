import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tree.naive_train.optimal_cut import find_cut_naive_given_discrete
from tree.optimized_train import optimized_train_tree
from tree.optimized_train.optimized_train_tree import get_top_by_scores, calculate_features_scores
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
        scores = calculate_features_scores(bins, y)
        naive_scores = np.array([find_cut_naive_given_discrete(y, x[:, i], 0)[1] for i in range(x.shape[1])])
        # Sine in partition to bins we lose information - we might miss the optimal split.
        # Therefore the naive scores are always a little bit better than the fast bin-based method.
        assert (scores > naive_scores -0.00001).all()
        error = (scores - naive_scores) / (naive_scores + scores)
        print('n_bins=', n_bins, 'max_error=', np.max(error))
        if min(v.bins_counts()) >= range_size:
            # Number of bins is at least as the number of values - binning does not lose
            # information.
            print('argmax', np.argmax(error))
            assert np.max(error) < 0.0001, n_bins


def test_train_tree_optimized():
    x = np.random.normal(size=(10000, 128))
    y = (x.T[7] > 0.1) * 5 + (x.T[2] < 0.01) * 3 + np.random.random(size=(x.shape[0])) * 0.01
    params = {'max_depth': 4, 'gamma': 0.1}
    rs_log2 = np.array([8, 4., 0])
    ks_log2 = np.array([0., 2, 4])
    params.update({'lines_sample_ratios': 2 ** -rs_log2,
                   'features_sample_ratios': 2 ** -ks_log2})
    tree = optimized_train_tree.train(x, y, params)
    print(tree.root())
    prediction = tree.predict_many(x)
    diff = prediction - y
    clean_diff = _drop_outliers(diff, 0.01)
    assert np.abs(clean_diff).max() < 0.1


def _drop_outliers(values, percent):
    assert 0 < percent < 1
    s = pd.Series(values)
    return s[(s >= s.quantile(percent)) &
             (s <= s.quantile(1 - percent))]