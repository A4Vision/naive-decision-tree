from typing import Tuple

import numpy as np

from tree.descision_tree import SimpleDecisionRule


def _validate_indices(rows_indices: np.ndarray, target_array_length: int):
    assert rows_indices.dtype in (np.int64, np.uint8)
    assert rows_indices.ndim == 1
    assert 0 <= rows_indices.max() < target_array_length


class NodeTrainDataView:
    def __init__(self, x: np.ndarray, y: np.ndarray, rows_indices: np.ndarray):
        assert y.shape[0] > 0
        assert y.shape[0] == x.shape[0]
        _validate_indices(rows_indices, x.shape[0])
        self._x = x
        self._y = y
        self._rows = rows_indices

    def all_rows(self):
        return self._rows

    def n_rows(self):
        return self._rows.shape[0]

    def k_features(self):
        return self._x.shape[1]

    def sample_rows(self, n):
        assert n <= self._rows.shape[0]
        return np.random.choice(self._rows, n, replace=False)

    def residue_values(self, rows: np.ndarray) -> np.ndarray:
        return self._y[rows]

    def features_values(self, features_list: np.ndarray, rows: np.ndarray) -> np.ndarray:
        _validate_indices(features_list, self.k_features())
        return self._x[np.ix_(rows, features_list)]

    def is_trivial_split(self, rule: SimpleDecisionRule) -> bool:
        decision = rule.decide_is_right_array(self._x[self._rows])
        return decision.sum() in (0, len(self._rows))

    def create_children_views(self, rule: SimpleDecisionRule) -> Tuple['NodeTrainDataView', 'NodeTrainDataView']:
        # TODO(Assaf): Does this operation copy the array, or does it just create a view ?
        decision = rule.decide_is_right_array(self._x[self._rows])
        left_rows = self._rows[~decision]
        right_rows = self._rows[decision]
        return NodeTrainDataView(self._x, self._y, left_rows), NodeTrainDataView(self._x, self._y, right_rows)