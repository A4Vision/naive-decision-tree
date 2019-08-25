import numpy as np

from tree.naive_train.optimal_cut import find_cut_naive_given_discrete
from tree.optimized_train.optimized_train_tree import get_top_by_scores, calculate_features_scores
from tree.optimized_train.value_to_bins import ValuesToBins


def test_take_top_k():
    top = get_top_by_scores(np.array([10, 2, 3, 4, 5]), np.array([0.1, -1, 2, 3, 4]), 2)
    assert set(top) == {10, 2}


def test_calculate_features_scores():
    y = np.random.random(10)
    range_size = 5
    np.random.seed(123)
    x = np.random.randint(13, 13 + range_size, size=(10, 30))
    for n_bins in (3, 4, 11):
        v = ValuesToBins(x, n_bins)
        bins = v.get_bins(x)
        scores = calculate_features_scores(bins, y)
        naive_scores = np.array([find_cut_naive_given_discrete(y, x[:, i], 0)[1] for i in range(x.shape[1])])
        assert np.min(scores - naive_scores) > -0.00001
        error = (scores - naive_scores) / (naive_scores + scores)
        print('n_bins=', n_bins, 'max_error=', np.max(error))
        if n_bins > range_size * 2:
            # We split by quantiles, so n_bins > range_size is not always enough.
            print('argmax', np.argmax(error))
            assert np.max(error) < 0.0001, n_bins
