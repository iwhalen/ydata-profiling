"""Test the numeric description functions for Ibis."""

import ibis
import numpy as np
import pandas as pd
import pytest
from pytest import approx

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.describe_counts_ibis import describe_counts_ibis
from ydata_profiling.model.ibis.describe_generic_ibis import describe_generic_ibis
from ydata_profiling.model.ibis.describe_numeric_ibis import describe_numeric_1d_ibis
from ydata_profiling.model.ibis.describe_supported_ibis import describe_supported_ibis
from ydata_profiling.model.pandas.describe_counts_pandas import pandas_describe_counts
from ydata_profiling.model.pandas.describe_generic_pandas import pandas_describe_generic
from ydata_profiling.model.pandas.describe_numeric_pandas import (
    pandas_describe_numeric_1d,
)
from ydata_profiling.model.pandas.describe_supported_pandas import (
    pandas_describe_supported,
)

test_data = [
    (
        pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name="int_data"),
        "integers_no_nulls",
    ),
    (
        pd.Series([1, 2, 3, 4, None, 6, 7, 8, None, 10], name="int_data_nulls"),
        "integers_with_nulls",
    ),
    (
        pd.Series(
            [1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5, 9.0, 10.5], name="float_data"
        ),
        "floats_no_nulls",
    ),
    (
        pd.Series(
            [1.0, None, 3.0, 4.5, 5.0, 6.5, None, 8.5, 9.0, None],
            name="float_data_nulls",
        ),
        "floats_nulls",
    ),
    (
        pd.Series(
            [1.0, np.nan, 3.0, 4.5, 5.0, 6.5, np.nan, 8.5, 9.0, np.nan],
            name="float_data_nans",
        ),
        "floats_nans",
    ),
    (
        pd.Series(
            [1.0, np.inf, 3.0, 4.5, 5.0, 6.5, np.inf, 8.5, 9.0, 10.5],
            name="float_data_inf",
        ),
        "floats_inf",
    ),
    (
        pd.Series(
            [1.0, -np.inf, 3.0, 4.5, 5.0, 6.5, -np.inf, 8.5, 9.0, 10.5],
            name="float_data_neg_inf",
        ),
        "floats_neg_inf",
    ),
    (
        pd.Series(
            [
                1.0,
                2.5,
                np.inf,
                4.5,
                None,
                -np.inf,
                np.nan,
                8.5,
                0,
                -10.5,
                -10.5,
            ],
            name="float_data_special",
        ),
        "floats_with_special_values",
    ),
    (pd.Series([0, 0, 0, 0], name="zeros_only"), "zeros_only"),
]


@pytest.mark.parametrize("series, test_id", test_data, ids=[d[1] for d in test_data])
def test_describe_numeric_ibis(series, test_id):
    config = Settings()
    config.plot.histogram.bins = 3

    ibis_table = ibis.memtable(series)

    summary_pandas = {}
    for func in [
        pandas_describe_counts,
        pandas_describe_generic,
        pandas_describe_supported,
        pandas_describe_numeric_1d,
    ]:
        _, _, summary_pandas = func(config, series, summary_pandas)

    summary_ibis = {}
    for func in [
        describe_counts_ibis,
        describe_generic_ibis,
        describe_supported_ibis,
        describe_numeric_1d_ibis,
    ]:
        _, _, summary_ibis = func(config, ibis_table, summary_ibis)

    keys = [
        "n_infinite",
        "p_infinite",
        "n_zeros",
        "p_zeros",
        "n_negative",
        "p_negative",
        "mad",
        "range",
        "mean",
        "variance",
        "std",
        "kurtosis",
        "skewness",
        "sum",
        "cv",
    ] + [f"{percentile:.0%}" for percentile in config.vars.num.quantiles]

    for key in keys:
        if np.isnan(summary_pandas[key]):
            assert np.isnan(
                summary_ibis[key]
            ), f'Key "{key}" is nan in pandas but not in ibis, got {summary_ibis[key]}'

        else:
            assert summary_ibis[key] == approx(
                summary_pandas[key]
            ), f'Key "{key}" not approximately equal'

    assert summary_ibis["monotonic"] == 0

    hist_ibis = summary_ibis["histogram"]
    hist_pandas = summary_pandas["histogram"]
    assert len(hist_ibis) == len(hist_pandas)

    # There is a bug in the Pandas implementation causing an all zero series to have only 1 bucket.
    # This test is skipped for this case.
    if not all(series == 0):
        np.testing.assert_array_equal(hist_ibis[0], hist_pandas[0])
        np.testing.assert_array_equal(hist_ibis[1], hist_pandas[1])
