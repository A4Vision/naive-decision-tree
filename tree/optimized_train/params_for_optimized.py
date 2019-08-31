"""
Optimized implementation of tree training.
1) Split the values of each feature into n_bins - according to the histogram.
2) Du
"""
from typing import Dict

import numpy as np


def _default_ratios():
    ks_log2 = np.array([0, 2, 4, 6, 8], dtype=np.float)
    rs_log2 = np.array([8, 6, 4, 2, 0], dtype=np.float)
    return {'lines_sample_ratios': 2 ** -ks_log2,
            'features_amounts': 2 ** -rs_log2}


def _set_defaults(params: Dict):
    params.setdefault('max_depth', 2)
    assert isinstance(params['max_depth'], int) and params['max_depth'] >= 0

    params.setdefault('n_bins', 256)
    # Make sure we can utilize uint8 for bins
    assert params['n_bins'] < 256

    params.setdefault('lines_sample_ratios', _default_ratios()['lines_sample_ratios'])
    params.setdefault('features_amounts', _default_ratios()['features_amounts'])

    assert params['lines_sample_ratios'] == params['features_amounts']


def print_expected_execution_statistics(params: Dict, n_samples: int, k_features: int):
    layer_runtime = sum(n_samples * params['lines_sample_ratios'] *
                        k_features * params['features_amounts'])
    naive_layer_runtime = n_samples * k_features
    speedup = naive_layer_runtime / layer_runtime
    print("Speedup:", speedup)
    print("Assuming lowest split contains N * 2 ** -(depth + 1) rows")
    n_rows_in_smallest_split = n_samples * (2 ** -(params['max_depth'] + 1))
    print("Expected amounts in lowest split:", n_rows_in_smallest_split * params['lines_sample_ratios'])
    print("features amounts:", k_features * params['features_amounts'])

