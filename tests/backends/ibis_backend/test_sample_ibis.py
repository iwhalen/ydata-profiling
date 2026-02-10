"""Test the sample functionality for Ibis backend."""

import ibis
import pandas as pd
import pytest

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.sample_ibis import get_sample_ibis
from ydata_profiling.model.pandas.sample_pandas import pandas_get_sample


@pytest.fixture
def data():
    return ibis.memtable(
        {
            "a": [1, 2, 3, 5, 7],
            "b": ["a", "b", "c", "d", "e"],
            "c": [True, False, True, False, True],
        }
    )


def test_ibis_get_sample(data):
    config = Settings()
    config.samples.head = 3
    config.samples.random = 0
    config.samples.tail = 0

    actual = get_sample_ibis(config, data)
    expected = pandas_get_sample(config, data.to_pandas())

    assert len(actual) == 1
    assert actual[0].id == "head"
    assert len(actual[0].data) == 3

    actual_data = actual[0].data
    expected_data = expected[0].data

    pd.testing.assert_frame_equal(actual_data, expected_data)


def test_ibis_get_sample_warnings(data):
    config = Settings()
    config.samples.head = 3
    config.samples.random = 1
    config.samples.tail = 1

    with pytest.warns(
        UserWarning, match="tail sample not implemented for Ibis"
    ), pytest.warns(UserWarning, match="random sample not implemented for Ibis"):
        get_sample_ibis(config, data)
