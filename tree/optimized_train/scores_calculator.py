import numpy as np

from tree.optimized_train.statistics_utils import ScoreEstimate, estimate_expectancy_of_sum_of_normal


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
            return self._calculate_estimate(values, i, scores_left[i], scores_right[i], confidence)

    def _calculate_estimate(self, values: np.ndarray, bin_value: int, score_left: float, score_right: float,
                            confidence: float) -> ScoreEstimate:
        # TODO(assaf): Remove the arguments score_left and score_right
        b = values <= bin_value
        left_y = self._y[b]
        left_estimate = estimate_expectancy_of_sum_of_normal(
            (left_y - np.average(left_y)) ** 2, confidence)
        right_y = self._y[~b]
        right_estimate = estimate_expectancy_of_sum_of_normal(
            (right_y - np.average(right_y)) ** 2, confidence)
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