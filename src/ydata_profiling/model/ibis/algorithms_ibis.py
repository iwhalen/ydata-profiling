from typing import Optional, Tuple

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

    clean = series.dropna()
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

    return bin_edges, hist_counts
