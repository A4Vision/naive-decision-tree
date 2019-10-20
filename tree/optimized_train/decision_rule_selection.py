import abc
from typing import Optional, List

import numpy as np

from tree.descision_tree import SimpleDecisionRule
from tree.naive_train import train_tree
from tree.naive_train.optimal_cut import calc_score, find_cut_naive_given_discrete
from tree.naive_train.train_tree import select_decision_rule
from tree.optimized_train.data_view import NodeTrainDataView
from tree.optimized_train.runtime_stats import RuntimeStats
from tree.optimized_train.scores_calculator import calculate_features_scores, ScoresCalculator

MINIMAL_ROWS_AMOUNT = 100


class DecisionRuleSelector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def select_decision_rule(self, data_view: NodeTrainDataView) -> Optional[SimpleDecisionRule]:
        pass


class ScheduledPruningSelector(DecisionRuleSelector):
    def __init__(self, features_sample_ratios: List[float],
                 lines_sample_ratios: List[float]):
        assert features_sample_ratios[0] == 1.
        assert lines_sample_ratios[-1] == 1.
        assert len(features_sample_ratios) == len(lines_sample_ratios)

        self._features_sample_ratios = list(features_sample_ratios)
        self._lines_sample_ratios = list(lines_sample_ratios)

    def select_decision_rule(self, data_view: NodeTrainDataView) -> Optional[SimpleDecisionRule]:
        current_features = np.arange(data_view.k_features())
        shifted_k_i = list(self._features_sample_ratios[1:]) + [1 / data_view.k_features()]

        for r_i, next_k_i in zip(self._lines_sample_ratios, shifted_k_i):
            rows_amount = int(np.ceil(r_i * data_view.n_rows()))
            if rows_amount < MINIMAL_ROWS_AMOUNT:
                continue
            rows = data_view.sample_rows(rows_amount)

            bins = data_view.features_values(current_features, rows)
            print('data-size', bins.shape)
            assert bins.shape == (rows.shape[0], len(current_features))
            y = data_view.residue_values(rows)
            features_scores = calculate_features_scores(bins, y)

            next_features_amount = int(np.round(next_k_i * data_view.k_features()))
            current_features = get_top_by_scores(current_features, features_scores, next_features_amount)
        all_rows = data_view.get_all_rows()

        temp_params = {}
        train_tree._set_defaults(temp_params)
        # TODO(Assaf): remove this dependency - extract the decision rule calculated in ScoresCalculator
        rule = select_decision_rule(data_view.features_values(current_features, all_rows),
                                    data_view.residue_values(all_rows), temp_params)
        return SimpleDecisionRule(rule.get_bound(), current_features[0])


def optimal_decision_rule_for_feature(best_feature_by_value, data_view) -> \
        Optional[SimpleDecisionRule]:
    all_rows = data_view.get_all_rows()
    x = data_view.features_values(np.array([best_feature_by_value]), all_rows)
    y = data_view.residue_values(all_rows)
    cut = naive_select_cut_discrete(x.T[0], y)
    if cut is None:
        return None
    return SimpleDecisionRule(cut, best_feature_by_value)


def naive_select_cut_discrete(x, y) -> Optional[int]:
    no_split_score = calc_score(y, 0)
    bound, score = find_cut_naive_given_discrete(y, x, 0)
    if score >= no_split_score:
        # Better not to split at this point
        return None
    else:
        return bound


class DynamicPruningSelector(DecisionRuleSelector):
    def __init__(self, lines_sample_ratios: List[float], confidence: float):
        self._confidence = confidence
        assert lines_sample_ratios[-1] == 1.

        self._lines_sample_ratios = list(lines_sample_ratios)
        self._runtime_stats = None

    def select_decision_rule(self, data_view: NodeTrainDataView) -> Optional[SimpleDecisionRule]:
        shape = (data_view.n_rows(), data_view.k_features())
        if self._runtime_stats is None:
            self._runtime_stats = RuntimeStats(shape)
        self._runtime_stats.start_decision_rule_calculation(shape)

        current_features = np.arange(data_view.k_features())
        best_feature_by_value = None

        for r_i in self._lines_sample_ratios:
            rows_amount = int(np.ceil(r_i * data_view.n_rows()))
            if rows_amount < MINIMAL_ROWS_AMOUNT:
                continue
            rows = data_view.sample_rows(rows_amount)
            bins = data_view.features_values(current_features, rows)
            print('data-size', bins.shape)
            self._runtime_stats.record_iteration(bins.shape)
            assert bins.shape == (rows.shape[0], len(current_features))

            y = data_view.residue_values(rows)
            scores_calc = ScoresCalculator(bins, y)
            feature_estimations = {feature: scores_calc.estimate_score(i, self._confidence)
                                   for i, feature in enumerate(current_features)}
            feature_estimations = {fe: score for fe, score in feature_estimations.items() if score.is_valid()}
            if not feature_estimations:
                break
            print('estimations of best', sorted(feature_estimations.values(), key=lambda x: x.value)[:5])
            lowest_upper_bound = min([estimation.upper_bound for estimation in feature_estimations.values()])
            current_features = np.array([feature for feature, estimation in feature_estimations.items() if
                                         estimation.lower_bound <= lowest_upper_bound])
            best_feature_by_value = min([(e.value, f) for f, e in feature_estimations.items()])[1]

        if best_feature_by_value is None:
            return None
        return optimal_decision_rule_for_feature(best_feature_by_value, data_view)

    def get_stats(self) -> RuntimeStats:
        return self._runtime_stats


def get_top_by_scores(values: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    return values[np.argsort(scores)][:k]


