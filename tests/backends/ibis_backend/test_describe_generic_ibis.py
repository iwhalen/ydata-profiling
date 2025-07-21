"""Tests the describe generic functionality for Ibis."""

import ibis
import pandas as pd
import pytest

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.describe_generic_ibis import describe_generic_ibis
from ydata_profiling.model.pandas.describe_generic_pandas import pandas_describe_generic


@pytest.fixture
def pandas_data() -> pd.DataFrame:
    data = {"A": [1, 2, 2, 3, 3, 3, None, None, 3, 1, 2]}
    return pd.DataFrame(data)


@pytest.fixture
def ibis_data(pandas_data: pd.DataFrame) -> ibis.Table:
    return ibis.memtable(pandas_data)


def test_describe_generic_compare(pandas_data, ibis_data):
    config = Settings()
    p_summary = {"n_missing": pandas_data["A"].isna().sum()}
    _, _, p_summary = pandas_describe_generic(config, pandas_data["A"], p_summary)

    i_summary = {"n_missing": pandas_data["A"].isna().sum()}
    _, _, i_summary = describe_generic_ibis(config, ibis_data.select("A"), i_summary)

    # Ibis does not implement memory_size
    p_summary.pop("memory_size")
    i_summary.pop("memory_size")

    assert p_summary == i_summary
