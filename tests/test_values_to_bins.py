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
