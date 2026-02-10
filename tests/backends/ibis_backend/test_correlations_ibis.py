import ibis
import pandas as pd
import pytest

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.correlations_ibis import (
    cramers_compute as ibis_cramers_compute,
)
from ydata_profiling.model.ibis.correlations_ibis import (
    kendall_compute as ibis_kendall_compute,
)
from ydata_profiling.model.ibis.correlations_ibis import (
    pearson_compute as ibis_pearson_compute,
)
from ydata_profiling.model.ibis.correlations_ibis import (
    phi_k_compute as ibis_phi_k_compute,
)
from ydata_profiling.model.ibis.correlations_ibis import (
    spearman_compute as ibis_spearman_compute,
)
from ydata_profiling.model.pandas.correlations_pandas import (
    pearson_compute as pandas_pearson_compute,
)


@pytest.fixture
def correlation_data_num():
    return ibis.memtable(
        {
            "test_num_1": [1, 2, 3, 5, 7, 8, 9],
            "test_num_2": [11, 12, 13, 15, 17, 18, 4],
        }
    )


@pytest.fixture
def correlation_var_types():
    return {"test_num_1": {"type": "Numeric"}, "test_num_2": {"type": "Numeric"}}


def test_pearson_ibis(correlation_data_num, correlation_var_types):
    cfg = Settings()

    res_ibis = ibis_pearson_compute(cfg, correlation_data_num, correlation_var_types)

    res_pandas = pandas_pearson_compute(cfg, correlation_data_num.to_pandas(), {})

    pd.testing.assert_frame_equal(res_pandas, res_ibis)


@pytest.mark.parametrize(
    "func",
    [
        ibis_spearman_compute,
        ibis_kendall_compute,
        ibis_phi_k_compute,
        ibis_cramers_compute,
    ],
)
def test_not_implemented(func, correlation_data_num):
    with pytest.raises(NotImplementedError):
        func(Settings(), correlation_data_num, {})
