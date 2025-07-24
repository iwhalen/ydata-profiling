"""Test the categorical describe function for Ibis."""

import ibis
import pandas as pd
import pytest

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.describe_categorical_ibis import (
    describe_categorical_1d_ibis,
)
from ydata_profiling.model.ibis.describe_counts_ibis import describe_counts_ibis
from ydata_profiling.model.ibis.describe_generic_ibis import describe_generic_ibis
from ydata_profiling.model.ibis.describe_supported_ibis import describe_supported_ibis
from ydata_profiling.model.pandas.describe_categorical_pandas import (
    pandas_describe_categorical_1d,
)
from ydata_profiling.model.pandas.describe_counts_pandas import pandas_describe_counts
from ydata_profiling.model.pandas.describe_generic_pandas import pandas_describe_generic
from ydata_profiling.model.pandas.describe_supported_pandas import (
    pandas_describe_supported,
)

TEST_DATA = [
    (
        pd.Series(["a", "b", "c", "a", "b", "c", "a", "b", "c"]),
        "character_categories",
    ),
    (
        pd.Series(["label_1", "label_2", "label_3", "label_1", "label_1", "label_3"]),
        "label_categories",
    ),
    (
        pd.Series(["a", "b", "c", "a", "b", "c", "a", "b", "c"], dtype="category"),
        "character_categories_as_category",
    ),
    (
        pd.Series(
            ["label_1", "label_2", "label_3", "label_1", "label_1", "label_3"],
            dtype="category",
        ),
        "label_categories_as_category",
    ),
    (
        pd.Series(["a", None, "c", "a", "b", None, "a", "b", "c"], dtype="category"),
        "character_categories_with_nulls_as_category",
    ),
    (
        pd.Series(
            [None, "label_2", "label_3", "label_1", "label_1", None],
            dtype="category",
        ),
        "label_categories_with_nulls_as_category",
    ),
]


@pytest.mark.parametrize("data, name", TEST_DATA)
def test_describe_categorical_1d_ibis(data, name):
    """Test that categorical description for Ibis matches the pandas version."""
    config = Settings()
    config.vars.text.length = True
    config.vars.text.characters = False
    config.vars.text.words = False
    config.plot.histogram.bins = 3

    table = ibis.memtable({"data": data})

    summary_ibis = {}
    for func in [
        describe_counts_ibis,
        describe_generic_ibis,
        describe_supported_ibis,
        describe_categorical_1d_ibis,
    ]:
        _, _, summary_ibis = func(config, table, summary_ibis)

    summary_pandas = {}
    for func in [
        pandas_describe_counts,
        pandas_describe_generic,
        pandas_describe_supported,
        pandas_describe_categorical_1d,
    ]:
        _, _, summary_pandas = func(config, data, summary_pandas)

    # Only check the imbalance.
    # The text summaries are tested in test_describe_text_ibis.
    assert summary_ibis["imbalance"] == pytest.approx(summary_pandas["imbalance"])
