"""Compute statistical description of Ibis datasets."""

from typing import Tuple

from ibis import Table
from tqdm import tqdm
from visions import VisionsTypeset

from ydata_profiling.config import Settings
from ydata_profiling.model.summarizer import BaseSummarizer
from ydata_profiling.utils.dataframe import sort_column_names

NUMERIC_TYPES = ()


def _get_type(series: Table) -> str:
    """Get the type of a series."""
    dtype = series.schema()[series.columns[0]]

    vtype = "Unsupported"

    if dtype.is_string():
        vtype = "Categorical"

    elif dtype.is_temporal():
        vtype = "DateTime"

    elif dtype.is_boolean():
        vtype = "Boolean"

    elif dtype.is_numeric():
        vtype = "Numeric"

    return vtype


def ibis_describe_1d(
    config: Settings,
    series: Table,
    summarizer: BaseSummarizer,
    typeset: VisionsTypeset,
) -> dict:
    """Describe a series (infer the variable type, then calculate type-specific values).

    Args:
        config: report Settings object
        series: The Series to describe.
        summarizer: Summarizer object
        typeset: Typeset

    Returns:
        A Series containing calculated series description values.
    """
    if config.infer_dtypes:
        raise NotImplementedError(
            "infer_dtypes is not supported for Ibis Tables. Please set infer_dtypes to False."
        )

    return summarizer.summarize(config, series, dtype=_get_type(series))  # type: ignore


def get_series_descriptions_ibis(
    config: Settings,
    df: Table,
    summarizer: BaseSummarizer,
    typeset: VisionsTypeset,
    pbar: tqdm,
) -> dict:
    """
    Compute series descriptions/statistics for a Spark DataFrame.

    Returns: A dict with the series descriptions for each column of a Dataset
    """

    def describe_column(name: str) -> Tuple[str, dict]:
        """Process a single Spark column using Spark's execution model."""
        description = ibis_describe_1d(config, df.select(name), summarizer, typeset)
        pbar.set_postfix_str(f"Describe variable: {name}")
        pbar.update()

        # Pop value counts to remove the reference.
        description.pop("value_counts", None)

        return name, description

    series_description = {col: describe_column(col)[1] for col in df.columns}

    # Sort and return descriptions
    return sort_column_names(series_description, config.sort)
