import abc
from typing import Optional, List

import numpy as np

from tree.descision_tree import SimpleDecisionRule
from tree.naive_train import train_tree
from tree.naive_train.train_tree import select_decision_rule
from tree.optimized_train.data_view import NodeTrainDataView
from tree.optimized_train.scores_calculator import calculate_features_scores, ScoresCalculator


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

        temp_params = {}
        train_tree._set_defaults(temp_params)
        # TODO(Assaf): remove this dependency - extract the decision rule calculated in ScoresCalculator
        rule = select_decision_rule(data_view.features_values(current_features, all_rows),
                                    data_view.residue_values(all_rows), temp_params)
        return SimpleDecisionRule(rule.get_bound(), current_features[0])


class DynamicPruningSelector(DecisionRuleSelector):
    def __init__(self, lines_sample_ratios: List[float]):
        assert lines_sample_ratios[-1] == 1.

        self._lines_sample_ratios = list(lines_sample_ratios)

    def select_decision_rule(self, data_view: NodeTrainDataView) -> Optional[SimpleDecisionRule]:
        current_features = np.arange(data_view.k_features())

        for r_i in self._lines_sample_ratios:
            rows_amount = int(np.ceil(r_i * data_view.n_rows()))
            rows = data_view.sample_rows(rows_amount)
            bins = data_view.features_values(current_features, rows)
            print('data-size', bins.shape)
            assert bins.shape == (rows.shape[0], len(current_features))

            y = data_view.residue_values(rows)
            scores_calc = ScoresCalculator(bins, y)

            feature_estimations = [(feature, scores_calc.estimate_score(feature, 0.95))
                                   for feature in current_features]
            lowest_upper_bound = min([estimation.upper_bound for feature, estimation in feature_estimations])
            current_features = [feature for feature, estimation in feature_estimations if
                                estimation.lower_bound <= lowest_upper_bound]
            print('len(current_features)', len(current_features), current_features)

        best_feature_by_value = min([(e.value, f) for f, e in feature_estimations])[1]
        current_features = np.array([best_feature_by_value])
        all_rows = np.arange(data_view.n_rows())

        temp_params = {}
        train_tree._set_defaults(temp_params)
        rule = select_decision_rule(data_view.features_values(current_features, all_rows),
                                    data_view.residue_values(all_rows), temp_params)
        return SimpleDecisionRule(rule.get_bound(), current_features[0])


def get_top_by_scores(values: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    return values[np.argsort(scores)][:k]
