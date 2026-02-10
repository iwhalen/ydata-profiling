import importlib

# Dynamically import all modules inside the Ibis folder
IBIS_MODULES = [
    "correlations_ibis",
    "dataframe_ibis",
    "describe_boolean_ibis",
    "describe_categorical_ibis",
    "describe_counts_ibis",
    "describe_date_ibis",
    "describe_generic_ibis",
    "describe_numeric_ibis",
    "describe_supported_ibis",
    "describe_text_ibis",
    "duplicates_ibis",
    "missing_ibis",
    "sample_ibis",
    "summary_ibis",
    "table_ibis",
    "timeseries_index_ibis",
]


# Load modules dynamically
for module_name in IBIS_MODULES:
    module = importlib.import_module(f"ydata_profiling.model.ibis.{module_name}")
    globals().update(
        {
            name: getattr(module, name)
            for name in dir(module)
            if not name.startswith("_")
        }
    )  # type: ignore

# Explicitly list all available functions
__all__: list[str] = [
    "describe_generic_ibis",
    "describe_boolean_1d_ibis",
    "describe_categorical_1d_ibis",
    "describe_text_1d_ibis",
    "describe_numeric_1d_ibis",
    "describe_date_1d_ibis",
    "describe_counts_ibis",
    "get_duplicates_ibis",
    "get_sample_ibis",
    "get_table_stats_ibis",
    "get_time_index_description_ibis",
    "get_series_descriptions_ibis",
]
