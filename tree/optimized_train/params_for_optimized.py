from typing import Dict

import numpy as np

from tree.optimized_train.decision_rule_selection import DynamicPruningSelector, DecisionRuleSelector

CONFIDENCE = 'confidence'
PRUNING_METHOD = 'feature_pruning_method'


def _default_ratios():
    ks_log2 = np.array([0, 2, 4], dtype=np.float)
    rs_log2 = np.array([4, 2, 0], dtype=np.float)
    return {'lines_sample_ratios': 2 ** -rs_log2,
            'features_sample_ratios': 2 ** -ks_log2}


def _set_defaults(params: Dict):
    # TODO(Assaf): Replace the params dict with a named tuple.
    params.setdefault(PRUNING_METHOD, DynamicPruningSelector)
    params.setdefault(CONFIDENCE, 0.7)
    assert 0 < params[CONFIDENCE] < 1
    assert isinstance(params[PRUNING_METHOD], DecisionRuleSelector)

    params.setdefault('max_depth', 2)
    assert isinstance(params['max_depth'], int) and params['max_depth'] >= 0

    params.setdefault('n_bins', 255)
    # Make sure we can utilize uint8 for bins
    assert params['n_bins'] < 256

    params.setdefault('lines_sample_ratios', _default_ratios()['lines_sample_ratios'])
    params.setdefault('features_sample_ratios', _default_ratios()['features_sample_ratios'])

    assert len(params['lines_sample_ratios']) == len(params['features_sample_ratios'])


def print_expected_execution_statistics(params: Dict, n_samples: int, k_features: int):
    layer_runtime = sum((n_samples * params['lines_sample_ratios']).astype(int) *
                        (k_features * np.array([1] + params['features_sample_ratios'][1:])).astype(int))
    naive_layer_runtime = n_samples * k_features
    speedup = naive_layer_runtime / layer_runtime
    print("Speedup:", speedup)
    print("Assuming lowest split contains N * 2 ** -(depth + 1) rows")
    n_rows_in_smallest_split = n_samples * (2 ** -(params['max_depth']))
    print("Expected amounts in lowest split:", n_rows_in_smallest_split * params['lines_sample_ratios'])
    print("features amounts:", k_features * params['features_sample_ratios'])

