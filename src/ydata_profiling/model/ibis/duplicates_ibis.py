from typing import Any, Dict, Optional, Sequence, Tuple

import ibis
import pandas as pd
from ibis import Table, _

from ydata_profiling.config import Settings
from ydata_profiling.model.duplicates import get_duplicates


@get_duplicates.register
def get_duplicates_ibis(
    config: Settings, df: Table, supported_columns: Sequence
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """Obtain the most occurring duplicate rows in the DataFrame.

    Args:
        config: report Settings object
        df: the Ibis Table.
        supported_columns: the columns to consider.

    Returns:
        A subset of the Table as a pandas DataFrame, ordered by occurrence.
    """
    n_head = config.duplicates.head
    duplicates_key = config.duplicates.key

    if duplicates_key in df.columns:
        raise ValueError(
            f"Duplicates key ({duplicates_key}) may not be part of the DataFrame. Either change the "
            f" column name in the DataFrame or change the 'duplicates.key' parameter."
        )

    metrics: Dict[str, Any] = {}
    out = None

    if n_head > 0:
        n_rows = df.count().execute()

        if supported_columns and n_rows > 0:
            duplicate_counts = (
                df.group_by(list(supported_columns))
                .aggregate(**{duplicates_key: _.count()})
                .filter(_[duplicates_key] > 1)
            )

            n_duplicates = duplicate_counts.count().execute()

            metrics["n_duplicates"] = n_duplicates
            metrics["p_duplicates"] = n_duplicates / n_rows

            out = (
                duplicate_counts.order_by(ibis.desc(duplicates_key))
                .limit(n_head)
                .to_pandas()
            )

        else:
            metrics["n_duplicates"] = 0
            metrics["p_duplicates"] = 0.0

    return metrics, out
