import numpy as np

from tree.descision_tree import SimpleDecisionRule
from tree.optimized_train.value_to_bins import ValuesToBins


def test_get_bins():
    x = np.array([[1, 2, 1, 2, 3, 3], [4, 4, 7, 4, 7, 7], [4, 4, 7, 0, 20, 7]]).T
    v = ValuesToBins(x, 3)
    bins = v.get_bins(x)
    assert bins.shape == x.shape
    assert bins.T.tolist() == [[0, 2, 0, 2, 3, 3], [0, 0, 2, 0, 2, 2], [1, 1, 2, 0, 3, 2]]


def test_conversion_rule():
    x = np.array([[1, 2, 1, 2, 3, 3], [4, 4, 7, 4, 7, 7], [4, 4, 7, 0, 20, 7]]).T
    v = ValuesToBins(x, 3)
    bins = v.get_bins(x)
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            bin_value = bins[row, col]
            rule_for_bins = SimpleDecisionRule(bin_value, col)
            rule_for_values = v.convert_to_values_rule(rule_for_bins)
            assert (rule_for_bins.decide_is_right_array(bins) ==
                    rule_for_values.decide_is_right_array(x)).all()


def _reduce_to_minimal_values(bins_values: np.ndarray):
    assert bins_values.ndim == 1
    sorted_unique_values = np.unique(bins_values).tolist()
    return np.array([sorted_unique_values.index(v) for v in bins_values])


def _assert_bins_equivalent(actual, expected):
    assert (_reduce_to_minimal_values(actual) == expected).all()


def test_binning_more_bins_than_values():
    x = np.array([[1, 1, 1, 2, 2], [1, 1, 2, 2, 3]]).T * 0.1
    v = ValuesToBins(x, 10)
    bins_t = v.get_bins(x).T
    _assert_bins_equivalent(bins_t[0], np.array([0, 0, 0, 1, 1]))
    _assert_bins_equivalent(bins_t[1], np.array([0, 0, 1, 1, 2]))


def test_numpy_does_not_crash():
    x = np.random.randint(0, 10, size=(1000, 2)) * 0.1
    b = ValuesToBins(x, 250)
    b.get_bins(x)
