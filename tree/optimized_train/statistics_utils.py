from typing import NamedTuple, Union

import numpy as np
import scipy.stats


class ScoreEstimate(NamedTuple):
    """
    Estimate of a score -
        value is an estimate of the score
        with high confidence, the score is in the range
            [lower_bound, upper_bound]
    """
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


def chunkify_padded(arr: np.ndarray, chunk_size: int) -> np.ndarray:
    """
    Pads the array with its mean so its length is a multiple of chunk_size,
    then splits the array to chunks of length chunk_size.
    """
    # TODO(shugybugy): optimize this point - we copy the whole array just to extend with average.
    arr = np.pad(arr, (0, arr.shape[0] % chunk_size), mode='mean')
    for i in range(arr.shape[0] // chunk_size):
        yield arr[i * chunk_size: (i + 1) * chunk_size]


def estimate_expectancy_of_sum_of_non_normal(non_normal_samples: np.ndarray,
                                             confidence: float, C: int = 10) -> ScoreEstimate:
    assert non_normal_samples.ndim == 1
    # Each sum of a chunk is approximately normal.
    new_samples = np.array([np.sum(chunk) for chunk in chunkify_padded(non_normal_samples, C)])
    return estimate_expectancy_of_sum_of_normal(new_samples, confidence)


def estimate_expectancy_of_sum_of_normal(samples: np.ndarray, confidence: float) -> ScoreEstimate:
    """
    :param samples: Independent samples of a normal variable.
    :return: An estimate for the expectancy of a sum of `samples.shape[0]` samples of X.
    """
    assert samples.ndim == 1
    return estimate_expectancy_of_normal_from_samples(samples, confidence) * samples.shape[0]


def estimate_expectancy_of_normal_from_samples(samples: np.ndarray, confidence: float) -> ScoreEstimate:
    """
    Estimates the mean of a normal variable X given samples of X.

    Gives a symmetric range around the "naive" estimation, which is average(X)

    :param samples: Independent samples of a normal variable X.
    :param confidence: A number between 0 to 1.
    The estimation is correct with the given confidence.
    I.e., if we define a normal variable X, and call this function 100000 times,
    with samples of X (different samples every time) and confidence=0.9 -
    then in 0.9 of the times, the real mean of X will be in the estimated output range.
    """
    assert samples.ndim == 1
    avg = np.average(samples)
    std = np.std(samples)
    n = samples.shape[0]
    alpha = 1 - confidence
    return ScoreEstimate(avg,
                         avg + scipy.stats.t.ppf(alpha / 2, n - 1) * std / n ** 0.5,
                         avg + scipy.stats.t.ppf(1 - alpha / 2, n - 1) * std / n ** 0.5)
