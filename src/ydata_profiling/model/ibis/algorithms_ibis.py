from typing import Optional, Tuple

import ibis
import numpy as np
from ibis import Table, _

from ydata_profiling.config import Settings


def histogram_compute(
    config: Settings, series: Table, column_name: Optional[str] = None
) -> Tuple[np.array, np.array]:
    """Calculate the histogram bins and values for a the given column.

    Note:
        This attempts to mimic the np.histogram function.

        See: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html

    Args:
        config: The settings object.
        series: The Ibis table used to compute the histogram.
        column_name: The name of the column to use in the histogram operation.
            If None, will use the first column of the table.

    Returns:
        A tuple containing the histogram bins and values.
    """
    if column_name is None:
        column_name = series.columns[0]

    bins = config.plot.histogram.bins
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(
            "Strictly positive integer number of histogram bins must be "
            f"provided for Ibis. Got {bins}."
        )

    clean = series.drop_null()
    if hasattr(clean[column_name], "isinf"):
        clean = clean.filter(~_[column_name].isinf())

    stats = (
        clean.aggregate(
            min_val=_[column_name].min(), max_val=_[column_name].max(), n=_.count()
        )
        .execute()
        .to_dict("records")[0]
    )
    min_val = stats["min_val"]
    max_val = stats["max_val"]
    n = stats["n"]

    if n == 0:
        return np.array([]), np.array([])

    # Match numpy histogram behavior for edge case.
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5

    # https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    hist_df = (
        clean.mutate(histogram=_[column_name].bucket(bin_edges, closed="left"))
        .group_by("histogram")
        .aggregate(
            hist=_.count(),
        )
        .order_by("histogram")
        .execute()
    )

    hist_counts = np.zeros(bins, dtype=int)
    for bucket, count in zip(hist_df["histogram"], hist_df["hist"]):
        if bucket == bins:
            hist_counts[bins - 1] += count
        else:
            hist_counts[bucket] += count

    return hist_counts, bin_edges


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


def entropy(table: Table, column_name: str, base: int = None) -> float:
    """Calculate the entropy of a column.

    Note:
        Follows the scipy.stats.entropy naming convention and behavior.

    Args:
        table: The Ibis table to calculate the entropy of.
        column_name: The name of the column to calculate the entropy of.
        base: The logarithmic base to use. If None, will use the natural logarithm.

    Returns:
        The entropy of the column.
    """
    # Most backends will ignore nans in the calculations, so we need to handle them here.
    if (
        hasattr(table[column_name], "isnan")
        and table.aggregate(
            null_or_nan=(table[column_name].isnan() | table[column_name].isnull()).any()
        )
        .execute()
        .iloc[0]
        .item()
    ):
        s = np.nan

    else:
        s = (
            table.mutate(pk=table[column_name] / table[column_name].sum())
            .mutate(
                summand=ibis.cases(
                    (_["pk"] > 0.0, -_["pk"] * _["pk"].log()),
                    (_["pk"] == 0.0, 0.0),
                    else_=float("-inf"),
                )
            )
            .aggregate(s=_["summand"].sum())
            .execute()
            .to_dict("records")[0]["s"]
        )

        if base is not None:
            s = (s / np.log(base)).item()

    return s


def column_imbalance_score(table: Table, column_name: str) -> float:
    """Calculate the imbalance score for a column.

    Args:
        table: The Ibis table containing the value counts.
        column_name: The name of the column to calculate the imbalance score for.

    Returns:
        The imbalance score for the column.
    """
    n_classes = table.count().execute()
    return (
        1 - (entropy(table, "count", base=2) / np.log2(n_classes))
        if n_classes > 1
        else 0
    )
