"""Test summary functionality for Ibis backend."""

from datetime import datetime

import ibis
import pytest
import tqdm

from ydata_profiling.config import IbisSettings
from ydata_profiling.model.ibis.summary_ibis import (
    _get_type,
    get_series_descriptions_ibis,
    ibis_describe_1d,
)
from ydata_profiling.model.summarizer import ProfilingSummarizer
from ydata_profiling.model.typeset import ProfilingTypeSet


@pytest.fixture
def data():
    return ibis.memtable(
        {
            "int_column": [1, 2, 3, 5, 7],
            "str_column": ["a", "b", "c", "d", "e"],
            "bool_column": [True, False, True, False, True],
            "float_column": [1.1, 2.2, 3.3, 4.4, 5.5],
            "date_column": [
                datetime(2021, 1, 1),
                datetime(2021, 1, 2),
                datetime(2021, 1, 3),
                datetime(2021, 1, 4),
                datetime(2021, 1, 5),
            ],
        }
    )


def test_get_series_descriptions_ibis(data):
    config = IbisSettings()

    typeset = ProfilingTypeSet(config)
    pbar = tqdm.tqdm()

    pbar = tqdm.tqdm(total=len(data.columns))
    actual = get_series_descriptions_ibis(
        config, data, ProfilingSummarizer(use_ibis=True, typeset=typeset), typeset, pbar
    )

    assert all(c in actual for c in data.columns)


@pytest.mark.parametrize(
    "column", ["int_column", "str_column", "bool_column", "float_column", "date_column"]
)
def test_ibis_describe_1d(data, column):
    config = IbisSettings()

    typeset = ProfilingTypeSet(config)

    actual = ibis_describe_1d(
        config,
        data.select(column),
        ProfilingSummarizer(use_ibis=True, typeset=typeset),
        typeset,
    )

    # Mostly want to be sure function run without errors.
    # So, we assert something that will be present in all outputs.
    assert "n" in actual


@pytest.mark.parametrize(
    "column, expected",
    [
        ("int_column", "Numeric"),
        ("str_column", "Categorical"),
        ("bool_column", "Boolean"),
        ("float_column", "Numeric"),
        ("date_column", "DateTime"),
    ],
)
def test__get_type(data, column, expected):
    actual = _get_type(data.select(column))

    assert actual == expected


def test__get_type_unsupported():
    actual = _get_type(ibis.memtable({"col": [[1, 2], [3], [4, 5, 6]]}))

    assert actual == "Unsupported"
