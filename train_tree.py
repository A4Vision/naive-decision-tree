import copy
from typing import Union

import numpy as np

from descision_tree import DecisionTree, LeafNode, SimpleDecisionRule, combine_two_trees
from optimal_cut import find_cut_naive, calc_score


def select_decision_rule(x, y, params) -> Union[SimpleDecisionRule, None]:
    scores_rules = []
    for feature_i in range(x.shape[1]):
        x_row = x.T[feature_i]
        argsort = np.argsort(x_row)
        y_sorted = y[argsort]
        x_row_sorted = x_row[argsort]
        i, score = find_cut_naive(y_sorted, params['gamma'])
        no_split_score = calc_score(y_sorted, params['gamma'])
        if score >= no_split_score:
            # Better not to split at this point
            scores_rules.append((score, None))
        else:
            scores_rules.append((score, SimpleDecisionRule(x_row_sorted[i - 1], feature_i)))
    best_score, best_rule = min(scores_rules, key=lambda x: x[0])
    return best_rule


def _set_defaults(params):
    params.setdefault('max_depth', 2)
    params.setdefault('gamma', 0.001)


def train(x, y, params) -> DecisionTree:
    assert y.shape[0] > 0
    params_copy = copy.deepcopy(params)
    _set_defaults(params_copy)
    assert isinstance(params_copy['max_depth'], int) and params_copy['max_depth'] >= 0
    if params['max_depth'] == 0 or y.shape[0] == 1:
        return DecisionTree(LeafNode(np.average(y)))
    else:
        params_copy['max_depth'] -= 1
        decision_rule = select_decision_rule(x, y, params)

        if decision_rule is None:
            return DecisionTree(LeafNode(np.average(y)))
        b_right = decision_rule.decide_is_right_array(x)
        b_left = ~b_right
        tree_right = train(x[b_right], y[b_right], params)
        tree_left = train(x[b_left], y[b_left], params)
        return combine_two_trees(decision_rule, tree_left, tree_right)
