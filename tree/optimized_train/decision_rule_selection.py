import abc
from typing import Optional, List, Dict, Iterable

import numpy as np

from tree.descision_tree import SimpleDecisionRule
from tree.naive_train.optimal_cut import calc_score, find_cut_naive_given_discrete
from tree.optimized_train.data_view import NodeTrainDataView
from tree.optimized_train.runtime_stats import RuntimeStats
from tree.optimized_train.scores_calculator import ScoresCalculator
from tree.optimized_train.statistics_utils import ScoreEstimate

MINIMAL_ROWS_AMOUNT = 100


class DecisionRuleSelector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def select_decision_rule(self, data_view: NodeTrainDataView) -> Optional[SimpleDecisionRule]:
        pass


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


class PruningSelector(DecisionRuleSelector):
    def __init__(self, lines_sample_ratios: List[float]):
        assert lines_sample_ratios[-1] == 1.

        self._lines_sample_ratios = list(lines_sample_ratios)
        self._runtime_stats = None

    def select_decision_rule(self, data_view: NodeTrainDataView) -> Optional[SimpleDecisionRule]:
        shape = (data_view.n_rows(), data_view.k_features())
        self._current_shape = shape
        if self._runtime_stats is None:
            self._runtime_stats = RuntimeStats(shape)
        self._runtime_stats.start_decision_rule_calculation(shape)

        current_features = np.arange(data_view.k_features())
        best_feature_by_value = None

        for index, r_i in enumerate(self._lines_sample_ratios):
            rows_amount = int(np.ceil(r_i * data_view.n_rows()))
            if rows_amount < MINIMAL_ROWS_AMOUNT:
                continue
            rows = data_view.sample_rows(rows_amount)
            bins = data_view.features_values(current_features, rows)
            self._runtime_stats.record_iteration(bins.shape)
            assert bins.shape == (rows.shape[0], len(current_features))

            y = data_view.residue_values(rows)
            scores_calc = ScoresCalculator(bins, y)
            feature_estimations = self._estimate_fetures_scores(current_features, scores_calc)
            feature_estimations = {fe: score for fe, score in feature_estimations.items() if score.is_valid()}
            if not feature_estimations:
                break
            current_features = self._keep_top_performing_features(feature_estimations, index)
            best_feature_by_value = min([(e.value, f) for f, e in feature_estimations.items()])[1]

        if best_feature_by_value is None:
            return None
        return optimal_decision_rule_for_feature(best_feature_by_value, data_view)

    @abc.abstractmethod
    def _keep_top_performing_features(self, feature_estimations: Dict[int, ScoreEstimate],
                                      iteration_index: int) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _estimate_fetures_scores(self, current_features: Iterable[int], scores_calc: ScoresCalculator) -> Dict[
        int, ScoreEstimate]:
        pass

    def get_stats(self) -> RuntimeStats:
        return self._runtime_stats


class DynamicPruningSelector(PruningSelector):
    def __init__(self, lines_sample_ratios: List[float], confidence: float):
        super().__init__(lines_sample_ratios)
        self._confidence = confidence

    def _keep_top_performing_features(self, feature_estimations: Dict[int, ScoreEstimate], iteration_index: int):
        lowest_upper_bound = min([estimation.upper_bound for estimation in feature_estimations.values()])
        return np.array([feature for feature, estimation in feature_estimations.items() if
                         estimation.lower_bound <= lowest_upper_bound])

    def _estimate_fetures_scores(self, current_features: Iterable[int], scores_calc: ScoresCalculator):
        return {feature: scores_calc.estimate_score(i, self._confidence)
                for i, feature in enumerate(current_features)}


class ScheduledPruningSelector(PruningSelector):
    def __init__(self, features_sample_ratios: List[float], lines_sample_ratios: List[float]):
        assert len(features_sample_ratios) == len(lines_sample_ratios)
        assert 0 < max(features_sample_ratios) < 1
        assert 0 < min(features_sample_ratios) < 1
        super().__init__(lines_sample_ratios)
        self._features_samples_ratios = list(features_sample_ratios)

    def _keep_top_performing_features(self, feature_estimations: Dict[int, ScoreEstimate], iteration_index: int):
        fs = list(feature_estimations.keys())
        k = int(self._current_shape[1] * self._features_samples_ratios[iteration_index])
        return get_top_by_scores(np.array([feature_estimations[f].value for f in fs]),
                                 np.array(fs), k)

    def _estimate_fetures_scores(self, current_features: Iterable[int], scores_calc: ScoresCalculator):
        return {feature: scores_calc.calculate_score(i)
                for i, feature in enumerate(current_features)}


def get_top_by_scores(values: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    return values[np.argsort(scores)][:k]


