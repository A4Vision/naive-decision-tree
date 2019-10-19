import re
from typing import List

import xgboost

LEAF_VALUE_PATTERN = re.compile("leaf=(.+)")


def _increase_leaf_value(line: str, offset: float):
    assert '\n' not in line
    for match in LEAF_VALUE_PATTERN.finditer(line):
        old_value = float(match.group(1))
        new_constant = old_value + offset
        new_line = line[:match.start(1)] + str(new_constant) + line[match.end(0):]
        return new_line
    return line


def increase_leaves_booster_text(dump: str, base_score: float):
    lines = [_increase_leaf_value(l, base_score) for l in dump.splitlines()]
    return '\n'.join(lines)


def booster_text(booster: xgboost.Booster, base_score: float):
    return increase_leaves_booster_text('\n'.join(booster.get_dump()),
                                        base_score)

