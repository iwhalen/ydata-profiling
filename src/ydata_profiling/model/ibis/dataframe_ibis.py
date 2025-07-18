import warnings

from ibis import Table
from ibis.expr.datatypes import (
    JSON,
    Array,
    DataType,
    LineString,
    Map,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Struct,
)

from ydata_profiling.config import Settings

IBIS_BAD_DTYPES = (
    Array,
    JSON,
    LineString,
    Map,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Struct,
)


def is_bad_type(dtype: DataType) -> bool:
    """Determine if the given dtype is a bad type."""
    return any(isinstance(dtype, t) for t in IBIS_BAD_DTYPES)


def ibis_preprocess(config: Settings, df: Table) -> Table:
    """Preprocess the Ibis DataFrame by removing the following column types:
    - Array
    - JSON
    - LineString
    - Map
    - MultiLineString
    - MultiPoint
    - MultiPolygon
    - Point
    - Polygon
    - Struct

    Args:
        config: Report settings object
        df: The Ibis Table

    Returns:
        The preprocessed Table without problematic columns.
    """

    # Identify MapType columns
    columns_to_remove = [
        (column, dtype) for column, dtype in df.schema().items() if is_bad_type(dtype)
    ]

    if columns_to_remove:
        warnings.warn(
            "Ibis Table profiling does not handle certain types. "
            f"The following columns will be ignored: {', '.join([column for column, _ in columns_to_remove])} "
            f"Due to their dtypes: {', '.join([str(dtype) for _, dtype in columns_to_remove])}."
        )
        df = df.drop(*[column for column, _ in columns_to_remove])

    return df
