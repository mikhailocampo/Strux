"""Strux: Structured Output Regression Testing for LLMs."""

from strux.configs import RegressionConfig, ValidationLevel
from strux.data_loading import DataSource, PostgresDataSource
from strux.pipeline import Pipeline, Sequential
from strux.strategies import exact_match, numeric_range, subset
from strux.results import RegressionResults

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Pipeline",
    "Sequential",
    "RegressionConfig",
    "RegressionResults",
    # Data Sources
    "DataSource",
    "PostgresDataSource",
    # Validation
    "ValidationLevel",
    # Built-in strategies
    "exact_match",
    "numeric_range",
    "subset",
]
