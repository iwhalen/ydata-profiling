"""Describe a categorical column."""

import numpy as np
from ibis import Table, _


def length_summary(series: Table) -> dict:
    """Describe the length of a categorical column.

    Args:
        series: The Ibis table to describe. Expected to be a single column.

    Returns:
        A dict containing the length statistics.
    """
    column_name = series.columns[0]

    n = series.count().execute()

    if n == 0:
        return {
            "max_length": np.nan,
            "mean_length": np.nan,
            "median_length": np.nan,
            "min_length": np.nan,
        }

    return (
        series.mutate(length=series[column_name].length())
        .aggregate(
            max_length=_.length.max(),
            mean_length=_.length.mean(),
            median_length=_.length.median(),
            min_length=_.length.min(),
        )
        .execute()
        .to_dict("records")[0]
    )


def unicode_summary(series: Table) -> dict:
    """Describe the character-level statistics of a text column.

    Args:
        series: The Ibis table to describe. Expected to be a single column.

    Returns:
        A dict containing the character-level statistics.
    """

    return {}
