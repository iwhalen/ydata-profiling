from typing import Tuple

from ibis import Table

from ydata_profiling.config import Settings
from ydata_profiling.model.ibis.algorithms_ibis import histogram_compute, length_summary


def describe_text_1d_ibis(
    config: Settings,
    series: Table,
    summary: dict,
) -> Tuple[Settings, Table, dict]:
    """Describe text column in an Ibis table.

    Args:
        config: report Settings object
        series: The single columned Ibis table to describe.
        summary: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
    column_name = series.columns[0]

    clean = series.drop_null()

    if not config.vars.cat.redact:
        summary.update({"first_rows": clean.limit(5).select(column_name).execute()})

    if config.vars.text.length:
        summary.update(**length_summary(clean))
        summary.update(
            **{
                "histogram_length": histogram_compute(
                    config,
                    clean.mutate(length=clean[column_name].length()),
                    column_name="length",
                )
            }
        )

    if config.vars.text.characters:
        raise NotImplementedError(
            "Character-level statistics are not implemented for Ibis"
        )

    if config.vars.text.words:
        raise NotImplementedError("Word-level statistics are not implemented for Ibis")

    return config, series, summary
