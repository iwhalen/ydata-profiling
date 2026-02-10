from typing import Tuple

import pandas as pd
from ibis import Table

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.algorithms_ibis import histogram_compute
from ydata_profiling.model.summary_algorithms import describe_date_1d


@describe_date_1d.register
def describe_date_1d_ibis(
    config: Settings, series: Table, summary: dict
) -> Tuple[Settings, Table, dict]:
    """Describe a single column table of dates.

    Args:
        config: report Settings object
        series: The Ibis table to describe.
        summary: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
    column_name = series.columns[0]

    if series[column_name].isnull().all().execute():
        summary.update({"min": pd.NaT, "max": pd.NaT, "range": 0, "histogram": []})

    else:
        summary.update(
            **series.aggregate(
                min=series[column_name].min(),
                max=series[column_name].max(),
            )
            .execute()
            .to_dict("records")[0]
        )

        if pd.isna(summary["min"]) or pd.isna(summary["max"]):
            summary["range"] = 0

        else:
            summary["range"] = summary["max"] - summary["min"]

        with_epoch = series.mutate(epoch=series[column_name].epoch_seconds())

        hist, bin_edges = histogram_compute(config, with_epoch, column_name="epoch")

        summary.update({"histogram": (hist, bin_edges)})

    invalid_dates = series.filter(series[column_name].isnull())
    n_invalid = invalid_dates.count().execute()
    summary.update(
        {
            "invalid_dates": invalid_dates.distinct().count().execute(),
            "n_invalid_dates": n_invalid,
            "p_invalid_dates": n_invalid / summary["n"] if summary["n"] > 0 else 0,
        }
    )

    return config, series, summary
