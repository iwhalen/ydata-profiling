"""Test missing plotting functionality for Ibis backend."""

import ibis
import numpy as np
import pytest

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.missing_ibis import missing_bar, missing_heatmap


@pytest.fixture
def data():
    return ibis.memtable(
        {
            "test_num_1": [1, 2, 3, 5, 7, 8, 10, 1],
            "test_num_2": [11, np.nan, 13, 15, 17, 18, 4, 11],
            "test_num_3": [11, np.nan, 13, 15, 17, 18, 4, 11],
            "test_num_4": [11, None, 13, 15, 17, 18, 4, 11],
            "test_num_5": [None, None, None, None, None, None, None, None],
        }
    )


@pytest.mark.parametrize("func", [missing_bar, missing_heatmap])
def test_no_errors(data, func):
    """Test that the function does not raise an error."""
    assert func(Settings(), data) is not None
