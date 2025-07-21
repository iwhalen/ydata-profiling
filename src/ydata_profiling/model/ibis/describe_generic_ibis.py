from typing import Tuple

import ibis

from ydata_profiling.config import Settings
from ydata_profiling.model.summary_algorithms import describe_generic


@describe_generic.register
def describe_generic_ibis(
    config: Settings, series: ibis.Table, summary: dict
) -> Tuple[Settings, ibis.Table, dict]:
    """Describe generic column of an Ibis Table.

    Args:
        config: report Settings object
        series: The colum to describe.
        summary: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """

    # number of observations in the Series
    length = series.count().execute()

    summary.update(
        {
            "n": length,
            "p_missing": summary["n_missing"] / length if length > 0 else 0,
            "count": length - summary["n_missing"],
            "memory_size": 0,  # TODO: what's the right way to calculate this at runtime?
        }
    )

    return config, series, summary
