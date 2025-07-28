"""Test the date description functions for Ibis."""

from datetime import datetime

import ibis
import pandas as pd
import pytest

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.describe_counts_ibis import describe_counts_ibis
from ydata_profiling.model.ibis.describe_date_ibis import describe_date_1d_ibis
from ydata_profiling.model.ibis.describe_generic_ibis import describe_generic_ibis
from ydata_profiling.model.ibis.describe_supported_ibis import describe_supported_ibis

test_data = [
    (
        pd.Series(
            [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)],
            name="datetime_data",
        ),
        "datetimes",
    ),
    (
        pd.Series(
            [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-02")],
            name="timestamp_data",
        ),
        "timestamps",
    ),
    (
        pd.Series(
            [datetime(2022, 1, 1), None, datetime(2022, 1, 3), None],
            name="datetime_data_nulls",
        ),
        "datetimes_with_none",
    ),
    (
        pd.Series(
            [datetime(2022, 1, 1), pd.NaT, datetime(2022, 1, 3), pd.NaT],
            name="datetime_data_nulls",
        ),
        "datetimes_with_NaT",
    ),
    (
        pd.Series(
            [pd.Timestamp("2022-01-01"), None, pd.Timestamp("2022-01-03"), None],
            name="timestamp_data_nulls",
        ),
        "timestamps_with_None",
    ),
    (
        pd.Series(
            [pd.Timestamp("2022-01-01"), pd.NaT, pd.Timestamp("2022-01-03"), pd.NaT],
            name="timestamp_data_nulls",
        ),
        "timestamps_with_NaT",
    ),
    (
        pd.Series(
            [datetime(2022, 1, 1), datetime(2022, 1, 1), datetime(2022, 1, 1)],
            name="datetimes_all_same",
        ),
        "datetimes_all_same",
    ),
    (
        pd.Series([], dtype="datetime64[ns]", name="empty_datetime_series"),
        "empty_datetime_series",
    ),
    (
        pd.Series([pd.NaT, pd.NaT, pd.NaT], name="all_nulls_series"),
        "all_nulls_series",
    ),
]


# TODO: Check expected behavior for Pandas backend.
# Right now, nulls are always removed before running pandas_describe_date_1d.
# This causes issues when counting invalid dates.
# For now, we'll compare to pandas functions directly instead of the backend.


@pytest.mark.parametrize("series, test_id", test_data, ids=[d[1] for d in test_data])
def test_describe_date_ibis(series, test_id):
    config = Settings()
    config.plot.histogram.bins = 3

    ibis_table = ibis.memtable(series.to_frame())

    summary = {}
    for func in [
        describe_counts_ibis,
        describe_generic_ibis,
        describe_supported_ibis,
        describe_date_1d_ibis,
    ]:
        _, _, summary = func(config, ibis_table, summary)

    if series.isna().all() or series.empty:
        assert pd.isna(summary["min"])
        assert pd.isna(summary["max"])
        assert summary["range"] == 0
        assert summary["invalid_dates"] == series[series.isna()].nunique(dropna=False)
        assert summary["n_invalid_dates"] == pd.isna(series).sum()
        assert summary["p_invalid_dates"] == (
            pd.isna(series).sum() / series.shape[0] if series.shape[0] > 0 else 0
        )

    else:
        assert summary["min"] == series.min()
        assert summary["max"] == series.max()
        assert summary["range"] == series.max() - series.min()
        assert summary["invalid_dates"] == series[series.isna()].nunique(dropna=False)
        assert summary["n_invalid_dates"] == series.isna().sum()
        assert summary["p_invalid_dates"] == series.isna().sum() / series.shape[0]
