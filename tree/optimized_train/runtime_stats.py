from typing import Tuple

import pandas as pd


class RuntimeStats:
    def __init__(self, all_data_shape: Tuple[int, int]):
        self._x_shape = []
        self._data_size = []
        self._current_x_shape = None
        self._data_shape = all_data_shape

    def as_dataframe(self) -> pd.DataFrame:
        res = pd.DataFrame({'x_shape': self._x_shape,
                            'data_size': self._data_size})
        res['runtime'] = res['data_size'].map(lambda x: x[0] * x[1])
        return res

    def start_decision_rule_calculation(self, x_shape):
        self._current_x_shape = x_shape

    def record_iteration(self, data_size: Tuple[int, int]):
        self._x_shape.append(self._current_x_shape)
        self._data_size.append(data_size)
