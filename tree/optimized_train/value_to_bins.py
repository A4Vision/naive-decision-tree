import numpy as np

from tree.descision_tree import DecisionTree, SimpleDecisionRule


class ValuesToBins:
    def __init__(self, x: np.ndarray, n_bins: int):
        assert n_bins < 256
        self._x = x
        self._max_values = self._x.max(axis=0)
        self._n_bins = n_bins
        self._quantiles = self._calculate_quantiles()
        assert self._quantiles.shape == (x.shape[1], n_bins)

    def _calculate_quantiles(self):
        return np.array([np.sort(np.percentile(v,
                              100 * np.arange(self._n_bins) / self._n_bins))
                         for v in self._x.T])

    def get_bins(self, x):
        assert x.shape[1] == self._x.shape[1]
        return np.array([np.digitize(x[:, i], self._quantiles[i], True) for i in range(self._x.shape[1])],
                        dtype=np.uint8).T

    def convert_to_values_rule(self, bins_rule: SimpleDecisionRule) -> SimpleDecisionRule:
        i = bins_rule.get_i()
        if bins_rule.get_bound() == self._n_bins:
            bound = self._max_values[i]
        else:
            bound = self._quantiles[i, bins_rule.get_bound()]
        return SimpleDecisionRule(bound, i)

    def convert_bins_tree_to_prediction_tree(self, tree_trained_on_bins: DecisionTree) -> DecisionTree:
        converted_root = tree_trained_on_bins.root().convert(self.convert_to_values_rule)
        return DecisionTree(converted_root)

    def bins_counts(self):
        return [len(set(self._quantiles[i])) for i in range(self._quantiles.shape[0])]
