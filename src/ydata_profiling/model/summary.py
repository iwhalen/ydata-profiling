"""Compute statistical description of datasets."""

from typing import Any

import pandas as pd
from tqdm import tqdm
from visions import VisionsTypeset

from ydata_profiling.config import Settings
from ydata_profiling.model.dataframe import ibisDataFrame, sparkDataFrame
from ydata_profiling.model.pandas.summary_pandas import (
    pandas_describe_1d,
    pandas_get_series_descriptions,
)
from ydata_profiling.model.summarizer import BaseSummarizer
from ydata_profiling.utils.backend import is_ibis_installed, is_pyspark_installed

if is_pyspark_installed():
    from ydata_profiling.model.spark.summary_spark import (  # noqa: E402    from ydata_profiling.model.spark.summary_spark import (  # noqa: E402
        get_series_descriptions_spark,
        spark_describe_1d,
    )

if is_ibis_installed():
    from ydata_profiling.model.ibis.summary_ibis import (  # noqa: E402
        get_series_descriptions_ibis,
        ibis_describe_1d,
    )


def describe_1d(
    config: Settings,
    series: Any,
    summarizer: BaseSummarizer,
    typeset: VisionsTypeset,
) -> dict:
    """
    Describe the given column with the appropriate backend.

    Args:
        config: report Settings object
        series: The Series to describe.
        summarizer: Summarizer object
        typeset: Typeset
    Returns:
    """
    if isinstance(series, pd.Series):
        return pandas_describe_1d(config, series, summarizer, typeset)
    elif isinstance(series, sparkDataFrame):  # type: ignore
        return spark_describe_1d(config, series, summarizer, typeset)
    elif isinstance(series, ibisDataFrame):  # type: ignore
        return ibis_describe_1d(config, series, summarizer, typeset)
    else:
        raise TypeError(f"Unsupported series type: {type(series)}")


def get_series_descriptions(
    config: Settings,
    df: Any,
    summarizer: BaseSummarizer,
    typeset: VisionsTypeset,
    pbar: tqdm,
) -> dict:
    if isinstance(df, pd.DataFrame):
        return pandas_get_series_descriptions(config, df, summarizer, typeset, pbar)
    elif isinstance(df, sparkDataFrame):  # type: ignore
        return get_series_descriptions_spark(config, df, summarizer, typeset, pbar)
    elif isinstance(df, ibisDataFrame):  # type: ignore
        return get_series_descriptions_ibis(config, df, summarizer, typeset, pbar)
    else:
        raise TypeError(f"Unsupported dataframe type: {type(df)}")
