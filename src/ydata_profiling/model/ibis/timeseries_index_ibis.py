"""Compute statistical description of datasets."""

import warnings

from ibis import Table

from ydata_profiling.config import Settings
from ydata_profiling.model.timeseries_index import get_time_index_description


@get_time_index_description.register
def get_time_index_description_ibis(
    config: Settings,
    df: Table,
    table_stats: dict,
) -> dict:
    warnings.warn("Time series index description not implemented for Ibis")
    return {}
