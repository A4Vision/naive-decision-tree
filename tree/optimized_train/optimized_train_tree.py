import copy
from typing import Dict

import numpy as np

from tree.descision_tree import DecisionTree, LeafNode, combine_two_trees
from tree.optimized_train.data_view import NodeTrainDataView
from tree.optimized_train.decision_rule_selection import DecisionRuleSelector, DynamicPruningSelector, \
    ScheduledPruningSelector
from tree.optimized_train.params_for_optimized import _set_defaults, print_expected_execution_statistics
from tree.optimized_train.value_to_bins import ValuesToBins


def _create_selector(params: Dict) -> DecisionRuleSelector:
    if params['feature_pruning_method'] == 'dynamic':
        return DynamicPruningSelector(params['lines_sample_ratios'])
    elif params['feature_pruning_method'] == 'scheduled':
        return ScheduledPruningSelector(params['features_sample_ratios'], params['lines_sample_ratios'])
    else:
        raise ValueError("Invalid pruning method: " + str(params['feature_pruning_method']))


def train(x, y, params) -> DecisionTree:
    assert y.shape[0] > 0
    assert y.shape[0] == x.shape[0]
    params_copy = copy.deepcopy(params)
    _set_defaults(params_copy)
    print_expected_execution_statistics(params_copy, x.shape[0], x.shape[1])
    converter = ValuesToBins(x, params_copy['n_bins'])
    binned_x = converter.get_bins(x)
    assert binned_x.dtype == np.uint8, binned_x.dtype
    assert binned_x.shape == x.shape
    binned_data_view = NodeTrainDataView(binned_x, y, np.arange(binned_x.shape[0]))
    tree = train_on_binned(binned_data_view, _create_selector(params_copy), params_copy)
    return converter.convert_bins_tree_to_prediction_tree(tree)


def train_on_binned(data_view: NodeTrainDataView, decision_rule_select: DecisionRuleSelector,
                    params: Dict) -> DecisionTree:
    params_copy = copy.copy(params)
    y = data_view.residue_values(data_view.all_rows())
    default_leaf_node = DecisionTree(LeafNode(np.average(y)))
    if params['max_depth'] == 0 or y.shape[0] == 1:
        return default_leaf_node
    else:
        params_copy['max_depth'] -= 1
        decision_rule = decision_rule_select.select_decision_rule(data_view)

        if decision_rule is None:
            return default_leaf_node
        if data_view.is_trivial_split(decision_rule):
            return default_leaf_node
        left_view, right_view = data_view.create_children_views(decision_rule)

        left_tree = train_on_binned(left_view, decision_rule_select, params_copy)
        right_tree = train_on_binned(right_view, decision_rule_select, params_copy)
        return combine_two_trees(decision_rule, left_tree, right_tree)
