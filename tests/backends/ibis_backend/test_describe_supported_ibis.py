import ibis
import pandas as pd
import pytest

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.describe_supported_ibis import describe_supported_ibis
from ydata_profiling.model.pandas.describe_supported_pandas import (
    pandas_describe_supported,
)


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({"A": [1, 2, 2, 3, 3, 3, 4, 5, None, None]}),
        pd.DataFrame({"A": [1, 2, 3, 4, 5]}),
        pd.DataFrame({"A": [1, 1, 2, 2]}),
        pd.DataFrame({"A": []}),
        pd.DataFrame({"A": [None, None]}),
    ],
)
def test_describe_supported_ibis_matches_pandas(data: pd.DataFrame):
    config = Settings()
    series = data["A"]
    ibis_table = ibis.memtable(data)
    ibis_series = ibis_table.select("A")

    # Manually construct pandas summary input
    pandas_summary_input = {
        "count": series.count(),
        "value_counts_without_nan": series.value_counts(),
        "hashable": True,
    }

    # Manually construct Ibis summary input
    ibis_summary_input = {
        "count": ibis_series["A"].count().execute(),
        "value_counts": ibis_series.value_counts(name="count"),
        "value_counts_without_nan": ibis_series.drop_null().value_counts(name="count"),
    }

    # Pandas path
    _, _, pandas_summary = pandas_describe_supported(
        config, series, pandas_summary_input
    )

    # Ibis path
    _, _, ibis_summary = describe_supported_ibis(
        config, ibis_series, ibis_summary_input
    )

    # Assertions
    assert ibis_summary["n_distinct"] == pandas_summary["n_distinct"]
    assert ibis_summary["p_distinct"] == pytest.approx(pandas_summary["p_distinct"])
    assert ibis_summary["is_unique"] == pandas_summary["is_unique"]
    assert ibis_summary["n_unique"] == pandas_summary["n_unique"]
    assert ibis_summary["p_unique"] == pytest.approx(pandas_summary["p_unique"])
