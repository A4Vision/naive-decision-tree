import numpy as np

from tree.optimized_train.statistics_utils import ScoreEstimate, estimate_expectancy_of_sum_of_normal, \
    estimate_expectancy_of_sum_of_non_normal


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

    def calculate_score(self, feature_i) -> ScoreEstimate:
        scores_sum = self._calculate_scores_per_cut(feature_i)
        score = np.min(scores_sum[np.isfinite(scores_sum)])
        return ScoreEstimate(score, score, score)

    def estimate_score(self, feature_i: int, confidence: float) -> ScoreEstimate:
        assert 0 <= feature_i < self._k
        assert 0 < confidence < 1
        scores_sum = self._calculate_scores_per_cut(feature_i)
        if not np.isfinite(scores_sum).any():
            return ScoreEstimate(np.inf, np.inf, np.inf)
        else:
            scores_sum[~np.isfinite(scores_sum)] = np.nan
            i = np.nanargmin(scores_sum)
            return self._calculate_estimate(self._bins[:, feature_i], i, confidence)

    def _calculate_scores_per_cut(self, feature_i):
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
        return scores_sum

    def _calculate_estimate(self, values: np.ndarray, bin_value: int, confidence: float) -> ScoreEstimate:
        b = values <= bin_value
        left_y = self._y[b]
        left_estimate = estimate_expectancy_of_sum_of_normal(
            (left_y - np.average(left_y)) ** 2, confidence)
        right_y = self._y[~b]
        right_estimate = estimate_expectancy_of_sum_of_normal(
            (right_y - np.average(right_y)) ** 2, confidence)
        return right_estimate + left_estimate


def calculate_features_scores(bins: np.ndarray, y: np.ndarray) -> np.ndarray:
    c = ScoresCalculator(bins, y)
    return np.array([c.calculate_score(feature_i).value
                     for feature_i in range(bins.shape[1])])
