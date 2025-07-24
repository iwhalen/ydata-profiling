"""Test duplicate counting functionality for Ibis."""

import ibis
import pandas as pd
import pytest

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.duplicates_ibis import get_duplicates_ibis
from ydata_profiling.model.pandas.duplicates_pandas import pandas_get_duplicates

TEST_DATA = [
    (
        pd.DataFrame(
            {
                "a": [1, 1, 4],
                "b": [1, 1, 2],
                "c": [3.1, 3.2, 3.3],
            }
        ),
        ["a", "b"],
        "duplicates",
    ),
    (
        pd.DataFrame(
            {
                "a": [1, 1, 4],
                "b": [1, 1, 2],
                "c": [3.1, 3.2, 3.3],
            }
        ),
        ["a", "b", "c"],
        "no_duplicates",
    ),
    (pd.DataFrame([{"a": None}]), ["a"], "empty"),
]


@pytest.mark.parametrize("head", [0, 1, 2, 5])
@pytest.mark.parametrize("data, supported_columns, name", TEST_DATA)
def test_duplicates_ibis(head, data, supported_columns, name):
    config = Settings()
    config.duplicates.head = head
    config.duplicates.key = "duplicate_key"

    metrics_expected, out_expected = pandas_get_duplicates(
        config, data, supported_columns
    )
    metrics_actual, out_actual = get_duplicates_ibis(
        config, ibis.memtable(data), supported_columns
    )

    if len(metrics_expected) > 0:
        assert metrics_actual["n_duplicates"] == metrics_expected["n_duplicates"]
        assert metrics_actual["p_duplicates"] == pytest.approx(
            metrics_expected["p_duplicates"]
        )
    else:
        assert metrics_actual == metrics_expected

    if out_expected is None:
        assert out_actual is None, f"Expected None, got a {type(out_actual)}"
    else:
        pd.testing.assert_frame_equal(out_actual, out_expected)
