"""Numeric description functions for Ibis."""

from typing import Tuple

import numpy as np
from ibis import Table, _

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.algorithms_ibis import histogram_compute
from ydata_profiling.model.summary_algorithms import describe_numeric_1d


def _kurtosis(t: Table) -> float:
    """Calculate the kurtosis of a numeric column."""
    # TODO: performance optimization: combine with skewness and reuse values from value_counts.
    column_name = t.columns[0]

    t_no_nan = t.filter(~t[column_name].isnull())
    count = t_no_nan.count().execute()

    if count < 4:
        return np.nan

    mean = t[column_name].sum().execute() / count

    moments = (
        t_no_nan.mutate(adjusted=t[column_name] - mean)
        .mutate(
            adjusted2=_.adjusted**2,
            adjusted4=_.adjusted**4,
        )
        .aggregate(m2=_.adjusted2.sum(), m4=_.adjusted4.sum())
        .execute()
    ).to_dict("records")[0]

    m2, m4 = moments["m2"], moments["m4"]

    # Copying behavior of pandas nankurt.
    # https://github.com/pandas-dev/pandas/blob/main/pandas/core/nanops.py
    with np.errstate(invalid="ignore", divide="ignore"):
        adj = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
        numerator = count * (count + 1) * (count - 1) * m4
        denominator = (count - 2) * (count - 3) * m2**2

    if denominator < 1e-14:
        denominator = 0

    if denominator == 0:
        return 0.0

    with np.errstate(invalid="ignore", divide="ignore"):
        result = numerator / denominator - adj

    return result


def _skewness(t: Table) -> float:
    """Calculate the skewness of a numeric column."""
    # TODO: performance optimization: combine with kurtosis and reuse values from value_counts.
    column_name = t.columns[0]

    t_no_nan = t.filter(~t[column_name].isnull())

    count = t_no_nan.count().execute()

    if count < 3:
        return np.nan

    mean = t[column_name].sum().execute() / count

    moments = (
        t_no_nan.mutate(adjusted=t[column_name] - mean)
        .mutate(
            adjusted2=_.adjusted**2,
            adjusted3=_.adjusted**3,
        )
        .aggregate(m2=_.adjusted2.sum(), m3=_.adjusted3.sum())
        .execute()
    ).to_dict("records")[0]

    m2, m3 = moments["m2"], moments["m3"]

    # Copying behavior of pandas nanskew.
    # https://github.com/pandas-dev/pandas/blob/main/pandas/core/nanops.py
    if m2 == 0:
        return 0.0

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (count * (count - 1) ** 0.5 / (count - 2)) * (m3 / m2**1.5)

    return result


@describe_numeric_1d.register
def describe_numeric_1d_ibis(
    config: Settings, series: Table, summary: dict
) -> Tuple[Settings, Table, dict]:
    """Describe a numeric column of an Ibis Table.

    Args:
        config: report Settings object
        series: The single column table to describe.
        summary: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
    col = series[series.columns[0]]

    stats = (
        series.aggregate(
            min=col.min(),
            max=col.max(),
            mean=col.mean(),
            sum=col.sum(),
        )
        .execute()
        .to_dict("records")[0]
    )
    summary.update(**stats)

    has_inf = False
    if hasattr(col, "isinf"):
        has_inf = col.isinf().any().execute()

    # Follow pandas behavior for inf values in std and variance calculations.
    if has_inf:
        summary["std"] = np.nan
        summary["variance"] = np.nan

    else:
        stats = (
            (series.filter(~col.isinf()) if has_inf else series)
            .aggregate(
                std=col.std(),
                variance=col.var(),
            )
            .execute()
            .to_dict("records")[0]
        )

        summary.update(**stats)

    summary["kurtosis"] = _kurtosis(series)
    summary["skewness"] = _skewness(series)

    if hasattr(col, "isinf"):
        n_infinite = col.isinf().sum().execute()
    else:
        n_infinite = 0

    summary["n_infinite"] = n_infinite
    summary["p_infinite"] = summary["n_infinite"] / summary["n"]
    summary["n_zeros"] = (col == 0).sum().execute()
    summary["p_zeros"] = summary["n_zeros"] / summary["n"]
    summary["n_negative"] = (col < 0).sum().execute()
    summary["p_negative"] = summary["n_negative"] / summary["n"]

    mad = (
        series.mutate(absolute_diff=(col - col.median()).abs())
        .aggregate(mad=_.absolute_diff.median())
        .execute()
        .to_dict("records")[0]["mad"]
    )

    summary["mad"] = mad

    summary["range"] = summary["max"] - summary["min"]

    percentiles = {
        f"{percentile:.0%}": value
        for percentile, value in zip(
            config.vars.num.quantiles,
            col.quantile(config.vars.num.quantiles).execute(),
        )
    }

    # Copy pandas behavior and swap infinities for nans.
    for key, value in percentiles.items():
        if np.isinf(value):
            percentiles[key] = np.nan

    summary.update(**percentiles)

    if "75%" in summary and "25%" in summary:
        summary["iqr"] = summary["75%"] - summary["25%"]
    else:
        q_25, q_75 = col.quantile([0.25, 0.75]).execute()

        summary["iqr"] = q_75 - q_25

    summary["cv"] = summary["std"] / summary["mean"] if summary["mean"] else np.nan

    # TODO: enable monotonicity check.
    # This would need an ordinal column to enforce ordering. Similar to Spark's limitation.
    summary["monotonic"] = 0

    summary.update({"histogram": histogram_compute(config, series)})

    return config, series, summary
