import numpy as np
import pytest

from tree.optimized_train.statistics_utils import estimate_mean_of_normal_from_samples


def test_estimate_normal():
    np.random.seed(123)
    avg = 20
    std = 3
    for n in [10, 1000]:
        for confidence in [0.8, 0.9, 0.95]:
            count_correct = 0
            N = 2 ** 9
            for _ in range(N):
                samples = np.random.normal(avg, std, size=n)
                estimate = estimate_mean_of_normal_from_samples(samples, confidence)
                is_estimation_correct = (estimate.lower_bound < avg < estimate.upper_bound)
                count_correct += int(is_estimation_correct)
            print(confidence, count_correct / N)
            assert pytest.approx(confidence, abs=0.03) == count_correct / N
