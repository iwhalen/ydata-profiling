"""
    File with a function to check the backend being used
"""

import importlib


def is_pyspark_installed() -> bool:
    """Check if PySpark is installed without importing it."""
    return importlib.util.find_spec("pyspark") is not None


def is_ibis_installed() -> bool:
    """Check if Ibis is installed without importing it."""
    return importlib.util.find_spec("ibis") is not None
