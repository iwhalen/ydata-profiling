"""Missing values plotting functionality for Ibis backend."""

import numpy as np
import pandas as pd
from ibis import Column, Table

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.correlations_ibis import pearson_compute
from ydata_profiling.visualisation.missing import plot_missing_bar, plot_missing_heatmap


def _is_null_or_nan(c: Column) -> Column:
    """Determine if a column is null or nan (if nan supported)."""
    out = c.isnull()
    if hasattr(c, "isnan"):
        out |= c.isnan()
    return out


def missing_bar(config: Settings, df: Table) -> str:
    """Generate missing values bar plot.

    Args:
        config: report Settings object
        df: Ibis Table.

    Returns:
        The resulting missing values bar plot encoded as a string.
    """
    null_counts = (
        df.aggregate(**{c: _is_null_or_nan(df[c]).sum() for c in df.columns})
        .to_pandas()
        .squeeze()
    )

    return plot_missing_bar(
        config,
        notnull_counts=null_counts,
        columns=df.columns,
        nrows=df.count().execute(),
    )


def missing_matrix(config: Settings, df: Table) -> str:
    """Not implemented."""
    raise NotImplementedError(
        "Missing matrix not implemented for Ibis backend. "
        "Set missing_diagrams.matrix=False in the config."
    )


def missing_heatmap(config: Settings, df: Table) -> str:
    """
    Generate missing values heatmap plot.

    Args:
        config: report Settings object
        df: Ibis Table.

    Returns:
        The resulting missing values heatmap plot encoded as a string.
    """
    # Exclude completely filled or empty columns.
    n = df.count().execute()

    null_matrix = df.mutate(**{c: _is_null_or_nan(df[c]) for c in df.columns})

    null_counts = (
        null_matrix.aggregate(**{c: null_matrix[c].sum() for c in null_matrix.columns})
        .to_pandas()
        .squeeze()
    )

    heatmap_columns = null_counts[(null_counts > 0) & (null_counts < n)].index.to_list()

    correlation_matrix: pd.DataFrame = pearson_compute(
        config, null_matrix, {c: {"type": "Numeric"} for c in heatmap_columns}
    )
    correlation_matrix = correlation_matrix.values

    mask = np.zeros_like(correlation_matrix)
    mask[np.triu_indices_from(mask)] = True

    return plot_missing_heatmap(
        config,
        corr_mat=correlation_matrix,
        mask=mask,
        columns=heatmap_columns,
    )
