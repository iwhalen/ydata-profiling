import ibis
import pandas as pd
import pytest

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.describe_counts_ibis import describe_counts_ibis
from ydata_profiling.model.pandas.describe_counts_pandas import pandas_describe_counts


@pytest.fixture
def pandas_data() -> pd.DataFrame:
    data = {"A": [1, 2, 2, 3, 3, 3, None, None, 3, 1, 2]}
    return pd.DataFrame(data)


@pytest.fixture
def ibis_data(pandas_data: pd.DataFrame) -> ibis.Table:
    return ibis.memtable(pandas_data)


def test_describe_counts_ibis_matches_pandas(
    pandas_data: pd.DataFrame, ibis_data: ibis.Table
):
    config = Settings()

    _, _, pandas_summary = pandas_describe_counts(config, pandas_data["A"], {})

    _, _, ibis_summary = describe_counts_ibis(config, ibis_data.select("A"), {})

    assert pandas_summary["n_missing"] == ibis_summary["n_missing"]
    assert pandas_summary["hashable"] == ibis_summary["hashable"]
    assert pandas_summary["ordering"] == ibis_summary["ordering"]

    pd.testing.assert_series_equal(
        pandas_summary["value_counts_without_nan"],
        ibis_summary["value_counts_without_nan"],
    )
