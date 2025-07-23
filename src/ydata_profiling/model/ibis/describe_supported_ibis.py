from typing import Tuple

from ibis import Table

from ydata_profiling.config import Settings
from ydata_profiling.model.summary_algorithms import describe_supported


@describe_supported.register
def describe_supported_ibis(
    config: Settings, series: Table, summary: dict
) -> Tuple[Settings, Table, dict]:
    """Describe a supported series.
    Args:
        series: The column of an Ibis Table to describe.
        series_description: The dict containing the series description so far.
    Returns:
        A dict containing calculated series description values.
    """
    count = summary["count"]
    value_counts = summary["value_counts"]

    distinct_count = value_counts.drop_null().count().execute()
    unique_count = value_counts.filter(value_counts["count"] == 1).count().execute()

    stats = {
        "n_distinct": distinct_count,
        "p_distinct": distinct_count / count if count > 0 else 0,
        "is_unique": unique_count == count and count > 0,
        "n_unique": unique_count,
        "p_unique": unique_count / count if count > 0 else 0,
    }
    stats.update(summary)

    return config, series, stats
