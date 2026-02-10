"""Table statistics for Ibis Tables."""

from collections import Counter

from ibis import Table

from ydata_profiling.config import Settings
from ydata_profiling.model.table import get_table_stats


@get_table_stats.register
def get_table_stats_ibis(config: Settings, df: Table, variable_stats: dict) -> dict:
    """General statistics for the Table.

    Args:
        config: report Settings object
        df: The Table
        variable_stats: Previously calculated statistics on the Table.

    Returns:
        A dictionary that contains the table statistics.
    """
    n = df.count().execute()

    table_stats = {
        "n": n,
        "n_var": len(df.columns),
        "n_cells_missing": 0,
        "n_vars_with_missing": 0,
        "n_vars_all_missing": 0,
    }

    for series_summary in variable_stats.values():
        if "n_missing" in series_summary and series_summary["n_missing"] > 0:
            table_stats["n_vars_with_missing"] += 1
            table_stats["n_cells_missing"] += series_summary["n_missing"]
            if series_summary["n_missing"] == n:
                table_stats["n_vars_all_missing"] += 1

    table_stats["p_cells_missing"] = (
        table_stats["n_cells_missing"] / (table_stats["n"] * table_stats["n_var"])
        if table_stats["n"] > 0 and table_stats["n_var"] > 0
        else 0
    )

    # Variable type counts
    table_stats.update(
        {"types": dict(Counter([v["type"] for v in variable_stats.values()]))}
    )

    return table_stats
