import copy
from typing import Tuple, Dict, Optional

import numpy as np

from tree.descision_tree import DecisionTree, SimpleDecisionRule, LeafNode, combine_two_trees
from tree.optimized_train import _set_defaults, print_expected_execution_statistics
from tree.naive_train.train_tree import select_decision_rule
from tree.optimized_train.value_to_bins import ValuesToBins


def _validate_indices(rows_indices: np.ndarray, target_array_length: int):
    assert rows_indices.dtype is np.int64
    assert rows_indices.ndim == 1
    assert 0 <= rows_indices.max() < target_array_length


class NodeTrainDataView:
    def __init__(self, x: np.ndarray, y: np.ndarray, rows_indices: np.ndarray):
        assert y.shape[0] > 0
        assert y.shape[0] == x.shape[0]
        _validate_indices(rows_indices, x.shape[0])
        self._x = x
        self._y = y
        self._rows = rows_indices

    def n_rows(self):
        return self._rows.shape[0]

    def k_features(self):
        return self._x.shape[1]

    def sample_rows(self, n):
        assert n <= self._rows.shape[0]
        return np.random.choice(self._rows, n, replace=False)

    def residue_values(self, rows: np.ndarray) -> np.ndarray:
        return self._y[rows]

    def features_values(self, features_list: np.ndarray, rows: np.ndarray) -> np.ndarray:
        _validate_indices(features_list, self.k_features())
        return self._x[rows, features_list]

    def create_children_views(self, rule: SimpleDecisionRule) -> Tuple['NodeTrainDataView', 'NodeTrainDataView']:
        decision = rule.decide_is_right_array(self._x[self._rows])
        left_rows = self._rows[~decision]
        right_rows = self._rows[decision]
        return NodeTrainDataView(self._x, self._y, left_rows), NodeTrainDataView(self._x, self._y, right_rows)


def train(x, y, params) -> DecisionTree:
    assert y.shape[0] > 0
    assert y.shape[0] == x.shape[0]
    params_copy = copy.deepcopy(params)
    _set_defaults(params_copy)
    print_expected_execution_statistics(params_copy, x.shape[0], x.shape[1])
    converter = ValuesToBins(x, params_copy['n_bins'])
    binned_x = converter.get_bins(x)
    assert binned_x.dtype is np.uint8
    binned_data_view = NodeTrainDataView(binned_x, y, np.arange(x.shape[0]))
    tree = train_on_binned(binned_data_view, params_copy)
    return converter.prediction_tree(tree)


def calculate_features_scores(bins: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert bins.shape[0] == y.shape[0]
    assert bins.ndim == 2
    assert y.ndim == 1
    total_sum = y.sum()
    y_sq = np.square(y)
    total_sum_sq = y_sq.sum()
    scores = []
    n, k = bins.shape
    for feature_i in range(bins.shape[1]):
        sums = np.bincount(bins[:, feature_i], weights=y)
        sums_sq = np.bincount(bins[:, feature_i], weights=y_sq)
        csum_sums = np.cumsum(sums)
        csum_sums_sq = np.cumsum(sums_sq)

        scores_left = csum_sums_sq - 1 / k * np.square(csum_sums)
        scores_right = (total_sum_sq - csum_sums_sq) - 1 / (n - k) * np.square(total_sum - csum_sums)
        scores.append(np.min(scores_left + scores_right))
    return np.array(scores)


def optimized_select_decision_rule(x_view: NodeTrainDataView, params: Dict) -> Optional[SimpleDecisionRule]:
    k_0 = params['features_amounts']
    assert k_0 == 1.
    r_t = params['lines_sample_ratios'][-1]
    assert r_t == 1.
    assert len(params['lines_sample_ratios']) == len(params['features_amounts'])
    current_features = np.arange(x_view.k_features())
    shifted_k_i = list(params['features_amounts'][1:]) + [1 / x_view.k_features()]
    for r_i, next_k_i in zip(params['lines_sample_ratios'], shifted_k_i):
        rows = x_view.sample_rows(int(np.ceil(r_i * x_view.n_rows())))
        bins = x_view.features_values(current_features, rows)
        assert bins.shape == (rows.shape[0], len(current_features))
        y = x_view.residue_values(rows)
        features_scores = calculate_features_scores(bins, y)
        next_features_amount = int(np.round(next_k_i * x_view.k_features()))
        current_features = get_top_by_scores(current_features, features_scores, next_features_amount)
    assert len(current_features) == 1
    all_rows = np.arange(x_view.n_rows())
    return select_decision_rule(x_view.features_values(current_features, all_rows),
                                x_view.residue_values(all_rows), params)


def get_top_by_scores(values: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    return values[np.argsort(scores)][:k]


def train_on_binned(x_view: NodeTrainDataView, params: Dict) -> DecisionTree:
    params_copy = copy.copy(params)

    if params['max_depth'] == 0 or y.shape[0] == 1:
        return DecisionTree(LeafNode(np.average(y)))
    else:
        params_copy['max_depth'] -= 1
        decision_rule = optimized_select_decision_rule(x_view, params)

        if decision_rule is None:
            return DecisionTree(LeafNode(np.average(y)))
        left_view, right_view = x_view.create_children_views(decision_rule)

        left_tree = train_on_binned(left_view, params_copy)
        right_tree = train_on_binned(right_view, params_copy)
        return combine_two_trees(decision_rule, left_tree, right_tree)
