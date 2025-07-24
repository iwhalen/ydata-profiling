"""Describe a boolean column."""

from typing import Tuple

import numpy as np
from ibis import Table

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.algorithms_ibis import column_imbalance_score


def describe_boolean_1d_ibis(
    config: Settings, series: Table, summary: dict
) -> Tuple[Settings, Table, dict]:
    """Describe a table with a single boolean column.

    Args:
        config: report Settings object.
        series: The table to describe.
        summary: The dict containing the series description so far.

    Returns:
        A dict containing calculated table description values.
    """
    value_counts = summary["value_counts"].drop_null()

    empty = value_counts.count().execute() == 0

    if not empty:
        top, freq = value_counts.head(1).to_pandas().iloc[0].to_list()
        summary.update({"top": top, "freq": freq})
        summary["imbalance"] = column_imbalance_score(value_counts, "count")

    else:
        summary.update(
            {
                "top": np.nan,
                "freq": 0,
                "imbalance": 0,
            }
        )

    return config, series, summary
