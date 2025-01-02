"""Strux: Structured Output Regression Testing for LLMs."""

from strux.configs import RegressionConfig, ValidationLevel
from strux.data_loading import DataSource, PostgresDataSource
from strux.pipeline import Pipeline, Sequential
from strux.results import RegressionResults
from strux.strategies import exact_match, numeric_range, subset

__version__ = "0.1.0"

__all__ = [
    "DataSource",
    "Pipeline",
    "PostgresDataSource",
    "RegressionConfig",
    "RegressionResults",
    "Sequential",
    "ValidationLevel",
    "exact_match",
    "numeric_range",
    "subset",
]
