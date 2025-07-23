"""Ibis counts."""

from typing import Tuple

import ibis
import pandas as pd
from ibis import Table, _  # noqa

from ydata_profiling.config import Settings
from ydata_profiling.model.summary_algorithms import describe_counts


@describe_counts.register
def describe_counts_ibis(
    config: Settings, series: Table, summary: dict
) -> Tuple[Settings, Table, dict]:
    """Counts the values in a series (with and without NaN, distinct).

    Args:
        config: Profiling settings.
        series: Ibis column for which we want to calculate the values.
        summary: Dictionary to store the summary results.

    Returns:
        Updated settings, input series, and summary dictionary.
    """
    column_name = series.columns[0]

    value_counts = (
        series.group_by(column_name)
        .aggregate(count=_.count())
        .order_by(ibis.desc("count"))
    )

    n_missing = series[column_name].isnull().sum().execute()
    if n_missing is None:
        n_missing = 0

    value_counts_no_nan = value_counts.filter(value_counts[column_name].notnull())

    def top_200_to_pandas(t: Table) -> pd.DataFrame:
        return t.limit(200).to_pandas().set_index(column_name)["count"]

    summary["n_missing"] = n_missing
    summary["value_counts"] = value_counts.cache()
    summary["value_counts_index_sorted"] = top_200_to_pandas(value_counts)
    summary["value_counts_without_nan"] = top_200_to_pandas(value_counts_no_nan)

    try:
        set(summary["value_counts_index_sorted"].index)
        hashable = True
    except ValueError:
        hashable = False

    summary["hashable"] = hashable
    summary["ordering"] = True

    return config, series, summary
