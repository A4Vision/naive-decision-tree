import copy
from typing import Dict, Optional

import numpy as np

from tree.descision_tree import DecisionTree, SimpleDecisionRule, LeafNode, combine_two_trees
from tree.naive_train.train_tree import select_decision_rule
from tree.optimized_train.data_view import NodeTrainDataView
from tree.optimized_train.params_for_optimized import _set_defaults, print_expected_execution_statistics
from tree.optimized_train.statistics_utils import estimate_sum_of_non_normal_samples_as_sum, ScoreEstimate, \
    estimate_sum_of_normal_samples_as_normal
from tree.optimized_train.value_to_bins import ValuesToBins


def train(x, y, params) -> DecisionTree:
    assert y.shape[0] > 0
    assert y.shape[0] == x.shape[0]
    params_copy = copy.deepcopy(params)
    _set_defaults(params_copy)
    print_expected_execution_statistics(params_copy, x.shape[0], x.shape[1])
    converter = ValuesToBins(x, params_copy['n_bins'])
    binned_x = converter.get_bins(x)
    assert binned_x.dtype == np.uint8, binned_x.dtype
    assert binned_x.shape == x.shape
    binned_data_view = NodeTrainDataView(binned_x, y, np.arange(binned_x.shape[0]))
    tree = train_on_binned(binned_data_view, params_copy)
    return converter.convert_bins_tree_to_prediction_tree(tree)


def optimized_select_decision_rule(data_view: NodeTrainDataView, params: Dict) -> Optional[SimpleDecisionRule]:
    k_0 = params['features_sample_ratios'][0]
    assert k_0 == 1., k_0
    r_t = params['lines_sample_ratios'][-1]
    assert r_t == 1.
    assert len(params['lines_sample_ratios']) == len(params['features_sample_ratios'])
    current_features = np.arange(data_view.k_features())
    shifted_k_i = list(params['features_sample_ratios'][1:]) + [1 / data_view.k_features()]

    for r_i, next_k_i in zip(params['lines_sample_ratios'], shifted_k_i):
        rows_amount = int(np.ceil(r_i * data_view.n_rows()))
        rows = data_view.sample_rows(rows_amount)

        bins = data_view.features_values(current_features, rows)
        print('data-size', bins.shape)
        assert bins.shape == (rows.shape[0], len(current_features))
        y = data_view.residue_values(rows)
        features_scores = calculate_features_scores(bins, y)

        next_features_amount = int(np.round(next_k_i * data_view.k_features()))
        current_features = get_top_by_scores(current_features, features_scores, next_features_amount)
        print('len(current_features)', len(current_features), current_features)
    assert len(current_features) == 1
    all_rows = np.arange(data_view.n_rows())
    rule = select_decision_rule(data_view.features_values(current_features, all_rows),
                                data_view.residue_values(all_rows), params)
    return SimpleDecisionRule(rule.get_bound(), current_features[0])


def new_optimized_optimized_select_decision_rule(data_view: NodeTrainDataView, params: Dict) -> Optional[SimpleDecisionRule]:
    r_t = params['lines_sample_ratios'][-1]
    assert r_t == 1.
    current_features = np.arange(data_view.k_features())

    for r_i in params['lines_sample_ratios']:
        rows_amount = int(np.ceil(r_i * data_view.n_rows()))
        rows = data_view.sample_rows(rows_amount)
        bins = data_view.features_values(current_features, rows)
        print('data-size', bins.shape)
        assert bins.shape == (rows.shape[0], len(current_features))

        y = data_view.residue_values(rows)
        scores_calc = ScoresCalculator(bins, y)

        feature_estimations = [(feature, scores_calc.estimate_score(feature, 0.95)) for feature in current_features]
        lowest_upper_bound = min([score.upper_bound for score in feature_estimations])
        current_features = [feature for feature, estimation in feature_estimations if estimation.lower_bound < lowest_upper_bound]
        print('len(current_features)', len(current_features), current_features)

    assert len(current_features) == 1
    all_rows = np.arange(data_view.n_rows())
    rule = select_decision_rule(data_view.features_values(current_features, all_rows),
                                data_view.residue_values(all_rows), params)
    return SimpleDecisionRule(rule.get_bound(), current_features[0])



def get_top_by_scores(values: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    return values[np.argsort(scores)][:k]


def train_on_binned(x_view: NodeTrainDataView, params: Dict) -> DecisionTree:
    params_copy = copy.copy(params)
    y = x_view.residue_values(x_view.all_rows())
    default_leaf_node = DecisionTree(LeafNode(np.average(y)))
    if params['max_depth'] == 0 or y.shape[0] == 1:
        return default_leaf_node
    else:
        params_copy['max_depth'] -= 1
        decision_rule = optimized_select_decision_rule(x_view, params)

        if decision_rule is None:
            return default_leaf_node
        if x_view.is_trivial_split(decision_rule):
            return default_leaf_node
        left_view, right_view = x_view.create_children_views(decision_rule)

        left_tree = train_on_binned(left_view, params_copy)
        right_tree = train_on_binned(right_view, params_copy)
        return combine_two_trees(decision_rule, left_tree, right_tree)


class ScoresCalculator:
    def __init__(self, bins: np.ndarray, y: np.ndarray):
        assert bins.shape[0] == y.shape[0]
        assert bins.ndim == 2
        assert y.ndim == 1
        self._y = y
        self._bins = bins
        self._total_sum = y.sum()
        self._y_sq = np.square(y)
        self._total_sum_sq = self._y_sq.sum()
        self._n, self._k = bins.shape

    def calculate_score(self, feature_i):
        return self.estimate_score(feature_i, 0.9).value

    def estimate_score(self, feature_i: int, confidence: float) -> ScoreEstimate:
        assert 0 <= feature_i < self._k
        assert 0 < confidence < 1
        # O(n)
        values = self._bins[:, feature_i]
        normal_bin_count = np.bincount(values)
        sums = np.bincount(values, weights=self._y)
        sums_sq = np.bincount(values, weights=self._y_sq)

        csum_sums = np.cumsum(sums)
        csum_sums_sq = np.cumsum(sums_sq)

        div_range = np.cumsum(normal_bin_count)
        assert div_range[-1] == self._n
        scores_left = csum_sums_sq - 1 / div_range * np.square(csum_sums)
        scores_right = (self._total_sum_sq - csum_sums_sq) - 1 / (self._n - div_range) * np.square(
            self._total_sum - csum_sums)
        scores_sum = scores_right + scores_left
        if not np.isfinite(scores_sum).any():
            return ScoreEstimate(np.inf, np.inf, np.inf)
        else:
            scores_sum[~np.isfinite(scores_sum)] = np.nan
            i = np.nanargmin(scores_sum)
            return self._calculate_estimate(values, i, scores_left[i], scores_right[i])

    def _calculate_estimate(self, values: np.ndarray, bin_value: int, score_left: float, score_right: float) -> ScoreEstimate:
        b = values <= bin_value
        left_y = self._y[b]
        left_estimate = estimate_sum_of_normal_samples_as_normal(
            (left_y - np.average(left_y)) ** 2, 0.9)
        right_y = self._y[~b]
        right_estimate = estimate_sum_of_normal_samples_as_normal(
            (right_y - np.average(right_y)) ** 2, 0.9)
        # print('length', len(left_y))
        # print('i', bin_value)
        # print('values', left_estimate.value, score_left)
        assert abs(left_estimate.value - score_left) < 0.0001
        assert abs(right_estimate.value - score_right) < 0.0001
        return right_estimate + left_estimate


def calculate_features_scores(bins: np.ndarray, y: np.ndarray) -> np.ndarray:
    c = ScoresCalculator(bins, y)
    return np.array([c.calculate_score(feature_i)
                     for feature_i in range(bins.shape[1])])


"""
We need to calculate on the left:
    SUM((avg_y - y) ** 2)), VARIANCE((avg_y - y) ** 2)


we have the sum of all the y
and we have the sum of their squares

we know the average of this random variable.
We need the average of its squared values.

So all we need is the sum of the squares.
That is, SUM((avg_y - y) ** 4)

"""
