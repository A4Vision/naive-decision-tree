import scipy.stats

from tree.optimized_train.optimized_train_tree import ScoreEstimate


def estimate_normal(avg: float, std: float, n: int, confidence: float) -> ScoreEstimate:
    alpha = 1 - confidence
    return ScoreEstimate(avg,
                         avg + scipy.stats.t.ppf(alpha / 2, n - 1) * std / n ** 0.5,
                         avg + scipy.stats.t.ppf(1 - alpha / 2, n - 1) * std / n ** 0.5)