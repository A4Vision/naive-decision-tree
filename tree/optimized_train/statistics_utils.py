from typing import NamedTuple, Union

import numpy as np
import scipy.stats


def chunkify_padded(arr: np.ndarray, size: int):
    # TODO(shugybugy): optimize this point - we copy the whole array just to extend with average.
    arr = np.pad(arr, (0, arr.shape[0] % size), mode='mean')
    for i in range(arr.shape[0] // size):
        yield arr[i * size: (i + 1) * size]


def estimate_sum_of_non_normal_samples_as_sum(non_normal_samples: np.ndarray, confidence: float, C=10):
    assert non_normal_samples.ndim == 1
    # Each sum of a chunk is approximately normal.
    new_samples = np.array([np.sum(chunk) for chunk in chunkify_padded(non_normal_samples, C)])
    return estimate_normal_from_samples(new_samples, confidence)


def estimate_sum_of_normal_samples_as_normal(samples: np.ndarray, confidence: float):
    assert samples.ndim == 1
    return estimate_normal_from_samples(samples, confidence) * samples.shape[0]


def estimate_normal_from_samples(samples: np.ndarray, confidence: float):
    assert samples.ndim == 1
    return estimate_normal(np.average(samples), np.std(samples), samples.shape[0], confidence)


class ScoreEstimate(NamedTuple):
    value: float
    lower_bound: float
    upper_bound: float

    def __mul__(self, number: Union[float, int]) -> 'ScoreEstimate':
        return ScoreEstimate(self.value * number, self.lower_bound * number, self.upper_bound * number)

    def __add__(self, other: Union[float, int, 'ScoreEstimate']) -> 'ScoreEstimate':
        if isinstance(other, ScoreEstimate):
            return ScoreEstimate(self.value + other.value,
                                 self.lower_bound + other.lower_bound,
                                 self.upper_bound + other.upper_bound)
        else:
            return ScoreEstimate(self.value + other,
                                 self.lower_bound + other,
                                 self.upper_bound + other)

    def __str__(self):
        return f"Estimate(mean={self.value},range=[{self.lower_bound}, {self.upper_bound}])"


def estimate_normal(avg: float, std: float, n: int, confidence: float) -> ScoreEstimate:
    alpha = 1 - confidence
    return ScoreEstimate(avg,
                         avg + scipy.stats.t.ppf(alpha / 2, n - 1) * std / n ** 0.5,
                         avg + scipy.stats.t.ppf(1 - alpha / 2, n - 1) * std / n ** 0.5)