"""Describe a categorical column."""

from typing import Tuple

import numpy as np
from ibis import Table, _

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.algorithms_ibis import entropy
from ydata_profiling.model.ibis.describe_text_ibis import describe_text_1d_ibis
from ydata_profiling.model.summary_algorithms import describe_categorical_1d


@describe_categorical_1d.register
def describe_categorical_1d_ibis(
    config: Settings, series: Table, summary: dict
) -> Tuple[Settings, Table, dict]:
    """
    Describe a categorical column.

    Args:
        config: The configuration object.
        series: The Ibis table to describe. Expected to be a single column.
        summarizer: The summarizer object.

    Returns:
        A dict containing the description of the categorical column.
    """
    config, series, summary = describe_text_1d_ibis(config, series, summary)

    value_counts = summary["value_counts"].drop_null()
    n_classes = value_counts.count().execute()
    summary["imbalance"] = (
        1 - (entropy(value_counts, "count", base=2) / np.log2(n_classes))
        if n_classes > 1
        else 0
    )

    return config, series, summary
