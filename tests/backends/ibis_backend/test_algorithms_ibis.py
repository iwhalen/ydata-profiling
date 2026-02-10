import ibis
import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.algorithms_ibis import entropy, histogram_compute

test_data = [
    (
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "integers_no_nulls",
    ),
    (
        np.array([1, 2, 3, 4, None, 6, 7, 8, None, 10]),
        "integers_with_nulls",
    ),
    (
        np.array([1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5, 9.0, 10.5]),
        "floats_no_nulls",
    ),
    (
        np.array([1.0, None, 3.0, 4.5, 5.0, 6.5, None, 8.5, 9.0, None]),
        "floats_nulls",
    ),
    (
        np.array([1.0, np.nan, 3.0, 4.5, 5.0, 6.5, np.nan, 8.5, 9.0, np.nan]),
        "floats_nans",
    ),
    (
        np.array([1.0, np.inf, 3.0, 4.5, 5.0, 6.5, np.inf, 8.5, 9.0, 10.5]),
        "floats_inf",
    ),
    (
        np.array([1.0, -np.inf, 3.0, 4.5, 5.0, 6.5, -np.inf, 8.5, 9.0, 10.5]),
        "floats_neg_inf",
    ),
    (
        np.array(
            [
                1.0,
                2.5,
                np.inf,
                4.5,
                None,
                -np.inf,
                np.nan,
                8.5,
                0,
                -10.5,
                -10.5,
            ]
        ),
        "floats_with_special_values",
    ),
    (np.array([0, 0, 0, 0]), "zeros_only"),
    (
        np.array(
            [
                -0.20547697,
                -2.08429223,
                -0.47450728,
                -1.06711806,
                -0.71782538,
                1.41754944,
                -1.54147593,
                -0.41826313,
                0.72911836,
                -0.973183,
                -0.27042909,
                -1.67609364,
                2.0821515,
                0.67926723,
                0.58568235,
                0.5415536,
                -0.72058637,
            ]
        ),
        "random_floats",
    ),
]


@pytest.mark.parametrize("bins", [3, 5, 10])
@pytest.mark.parametrize("data, name", test_data)
def test_histogram_compute(bins, data, name):
    config = Settings()
    config.plot.histogram.bins = bins

    column_name = "test_column"
    tbl = ibis.memtable({column_name: data})

    ibis_counts, ibis_bins = histogram_compute(
        config, series=tbl, column_name=column_name
    )

    if data.dtype == object:
        # Filter out None values and convert to float
        clean_data = data[data != np.array(None)].astype(float)
    else:
        clean_data = data.astype(float)

    # Filter out NaN and inf values
    clean_data = clean_data[np.isfinite(clean_data)]

    if len(clean_data) == 0:
        np_counts, np_bins = np.array([]), np.array([])
    else:
        np_counts, np_bins = np.histogram(clean_data, bins=config.plot.histogram.bins)

    assert len(ibis_counts) == len(np_counts)
    assert len(ibis_bins) == len(np_bins)
    np.testing.assert_array_equal(ibis_counts, np_counts)
    np.testing.assert_array_equal(ibis_bins, np_bins)


@pytest.mark.parametrize("base", [2, 10, None])
@pytest.mark.parametrize("data, name", test_data)
def test_entropy(base, data, name):
    # Scipy doesn't handle None, so we don't either.
    data = data[data != None]  # noqa: E711

    tbl = ibis.memtable({"data": data})

    actual = entropy(tbl, "data", base=base)
    # Scipy can't handle integers, so we cast to float.
    expected = scipy_entropy(data.astype(float), base=base)

    if np.isnan(expected):
        assert np.isnan(actual), f"Expected value is nan, actual is {actual}"

    else:
        assert actual == pytest.approx(expected)
