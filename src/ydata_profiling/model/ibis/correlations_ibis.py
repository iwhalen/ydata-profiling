"""Correlations between variables."""

from typing import Optional

import numpy as np
import pandas as pd
from ibis import Table

from ydata_profiling.config import Settings


def pearson_compute(
    config: Settings, df: Table, summary: dict
) -> Optional[pd.DataFrame]:
    """
    Compute Pearson correlation matrix for numeric columns.
    """
    num_cols = [col for col, desc in summary.items() if desc.get("type") == "Numeric"]

    if not num_cols or len(num_cols) < 2:
        matrix = np.array([])
        num_cols = []

    else:
        numeric_table = df[num_cols]

        matrix = np.empty([len(num_cols), len(num_cols)], dtype=float)

        for i, col1 in enumerate(num_cols):
            for j, col2 in enumerate(num_cols):
                if i <= j:
                    corr_value = (
                        numeric_table[col1]
                        .corr(numeric_table[col2], how="pop")
                        .execute()
                    )
                    matrix[i, j] = corr_value
                    matrix[j, i] = corr_value

    return pd.DataFrame(matrix, index=num_cols, columns=num_cols)


def spearman_compute(
    config: Settings, df: Table, summary: dict
) -> Optional[pd.DataFrame]:
    raise NotImplementedError()


def kendall_compute(
    config: Settings, df: Table, summary: dict
) -> Optional[pd.DataFrame]:
    raise NotImplementedError()


def cramers_compute(
    config: Settings, df: Table, summary: dict
) -> Optional[pd.DataFrame]:
    raise NotImplementedError()


def phi_k_compute(config: Settings, df: Table, summary: dict) -> Optional[pd.DataFrame]:
    raise NotImplementedError()
