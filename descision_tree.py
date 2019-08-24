from abc import abstractmethod
from typing import Callable

import numpy as np


class Node:
    @abstractmethod
    def is_leaf(self) -> bool:
        pass

    @abstractmethod
    def predict(self, vals) -> float:
        pass

    @abstractmethod
    def convert(self, conversion_rule) -> 'Node':
        pass


class LeafNode(Node):
    def __init__(self, value):
        self._value = value

    def predict(self, vals) -> float:
        return self._value

    def is_leaf(self) -> bool:
        return True

    def convert(self, conversion_rule):
        return LeafNode(self._value)


class SimpleDecisionRule:
    def __init__(self, bound, i):
        self._i = i
        self._bound = bound

    def decide_is_right(self, vals):
        assert vals.ndim == 1
        return self._bound < vals[self._i]

    def decide_is_right_array(self, values: np.array) -> np.array:
        assert values.ndim == 2
        return self._bound < values.T[self._i]

    def __str__(self):
        return f"(x[{self._i}] < {self._bound:.4f})"

    def get_bound(self):
        return self._bound

    def get_i(self):
        return self._i


class DecisionNode(Node):
    def __init__(self, decision_rule: SimpleDecisionRule):
        self._decision_rule = decision_rule
        self._left = None
        self._right = None

    def set_left(self, left: Node):
        self._left = left

    def set_right(self, right: Node):
        self._right = right

    def is_leaf(self):
        return False

    def predict(self, vals):
        if self._is_right(vals):
            return self._right.predict(vals)
        else:
            return self._left.predict(vals)

    def _is_right(self, vals):
        return self._decision_rule.decide_is_right(vals)

    def convert(self, conversion_rule: Callable[[SimpleDecisionRule], SimpleDecisionRule]) -> 'DecisionNode':
        new_rule = conversion_rule(self._decision_rule)
        res = DecisionNode(new_rule)
        res.set_right(self._right.convert(conversion_rule))
        res.set_left(self._left.convert(conversion_rule))
        return res


class DecisionTree:
    def __init__(self, root_node: Node):
        self._root = root_node

    def predict(self, vals):
        return self._root.predict(vals)

    def predict_many(self, vals: np.array):
        assert vals.ndim == 2
        return list(map(self.predict, vals))

    def root(self):
        return self._root


def combine_two_trees(root_decision_rule: SimpleDecisionRule,
                      tree_left: DecisionTree,
                      tree_right: DecisionTree) -> DecisionTree:
    root = DecisionNode(root_decision_rule)
    root.set_left(tree_left.root())
    root.set_right(tree_right.root())
    return DecisionTree(root)



