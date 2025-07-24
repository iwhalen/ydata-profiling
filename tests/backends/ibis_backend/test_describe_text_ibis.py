"""Test text description for Ibis backend."""

import ibis
import numpy as np
import pandas as pd
import pytest
from ibis.expr.datatypes import String

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.describe_text_ibis import describe_text_1d_ibis
from ydata_profiling.model.pandas.describe_counts_pandas import pandas_describe_counts
from ydata_profiling.model.pandas.describe_generic_pandas import pandas_describe_generic
from ydata_profiling.model.pandas.describe_supported_pandas import (
    pandas_describe_supported,
)
from ydata_profiling.model.pandas.describe_text_pandas import pandas_describe_text_1d

SENTENCES = [
    "The purple elephant danced gracefully on the moonlit rooftop.",
    "My coffee mug started speaking French after I added too much sugar.",
    "The library books whispered secrets to each other at midnight.",
    "The subway train conductor was actually three raccoons in a trench coat.",
    "My neighbor's garden gnome has been plotting to take over the neighborhood.",
]

SENTENCES_WITH_NULLS = SENTENCES[:2] + [None] + SENTENCES[2:] + [None]

SENTENCES_WITH_EMPTIES = SENTENCES[:2] + [""] + SENTENCES[2:] + ["  ", " "]

SENTENCES_WITH_MIXED_VALUES = SENTENCES_WITH_NULLS + SENTENCES_WITH_EMPTIES


def test_raises_not_implemented_error():
    """Test that the function raises a NotImplementedError for character-level statistics."""
    table = ibis.memtable({"text": SENTENCES})

    config = Settings()
    config.vars.text.characters = True
    config.vars.text.words = False
    config.vars.text.length = False

    with pytest.raises(NotImplementedError):
        describe_text_1d_ibis(config, table, {})

    config.vars.text.characters = False
    config.vars.text.words = True
    config.vars.text.length = False

    with pytest.raises(NotImplementedError):
        describe_text_1d_ibis(config, table, {})


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(SENTENCES, id="sentences"),
        pytest.param(SENTENCES_WITH_NULLS, id="sentences_with_nulls"),
        pytest.param(SENTENCES_WITH_EMPTIES, id="sentences_with_empties"),
        pytest.param(SENTENCES_WITH_MIXED_VALUES, id="sentences_with_mixed_values"),
        pytest.param([], id="empty"),
        pytest.param([None], id="single_null"),
        pytest.param([None, None, None], id="three_nulls"),
        pytest.param([""], id="single_empty_str"),
        pytest.param(["", "", ""], id="three_empty_str"),
    ],
)
def test_describe_text_1d_ibis(text):
    """Test that the length summary is correct."""
    series = pd.Series(text, dtype=str)
    table = ibis.memtable({"text": series}, schema={"text": String()})

    config = Settings()
    config.vars.text.length = True
    config.vars.text.characters = False
    config.vars.text.words = False
    config.plot.histogram.bins = 3

    summary_pandas = {}
    for func in [
        pandas_describe_counts,
        pandas_describe_generic,
        pandas_describe_supported,
        pandas_describe_text_1d,
    ]:
        _, _, summary_pandas = func(config, series, summary_pandas)

    _, _, summary_ibis = describe_text_1d_ibis(config, table, {})

    keys = ["max_length", "mean_length", "median_length", "min_length"]

    for key in keys:
        if np.isnan(summary_pandas[key]):
            assert np.isnan(
                summary_ibis[key]
            ), f'Key "{key}" is nan in pandas but not in ibis, got {summary_ibis[key]}'

        else:
            assert summary_ibis[key] == pytest.approx(
                summary_pandas[key]
            ), f'Key "{key}" not approximately equal'
