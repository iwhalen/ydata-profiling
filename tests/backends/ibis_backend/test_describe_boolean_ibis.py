"""Test boolean description for Ibis."""

import ibis
import numpy as np
import pandas as pd
import pytest

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.describe_boolean_ibis import describe_boolean_1d_ibis
from ydata_profiling.model.ibis.describe_counts_ibis import describe_counts_ibis
from ydata_profiling.model.pandas.describe_boolean_pandas import (
    pandas_describe_boolean_1d,
)
from ydata_profiling.model.pandas.describe_counts_pandas import pandas_describe_counts

TEST_DATA = [
    (
        pd.Series([True, False, True, False, True, False]),
        "booleans",
    ),
    (
        pd.Series([]),
        "empty",
    ),
    (
        pd.Series([None, None, None]),
        "nulls",
    ),
    (
        pd.Series([True, None, True, False, True, False, None]),
        "booleans_with_nulls",
    ),
]


@pytest.mark.parametrize("data, name", TEST_DATA)
def test_describe_boolean_1d_ibis(data, name):
    config = Settings()

    series = pd.Series(data)
    table = ibis.memtable({"data": series}, schema={"data": "boolean"})

    _, _, summary_ibis = describe_counts_ibis(config, table, {})
    _, _, summary_ibis = describe_boolean_1d_ibis(config, table, summary_ibis)

    _, _, summary_pandas = pandas_describe_counts(config, series, {})
    _, _, summary_pandas = pandas_describe_boolean_1d(config, series, summary_pandas)

    assert summary_ibis["freq"] == summary_pandas["freq"]
    assert summary_ibis["imbalance"] == pytest.approx(summary_pandas["imbalance"])
    if np.isnan(summary_pandas["top"]):
        assert np.isnan(summary_ibis["top"])
    else:
        assert summary_ibis["top"] == summary_pandas["top"]
