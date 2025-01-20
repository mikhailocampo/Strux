"""Strux: Structured Output Regression Testing for LLMs."""

from strux.configs import RegressionConfig, ValidationLevel
from strux.data_loading import DataSource, PostgresDataSource, CSVDataSource
from strux.pipeline import Pipeline, Sequential
from strux.results import RegressionResults
from strux.strategies import exact_match, subset, absolute_deviation, relative_deviation
from strux.experiment import Experiment
from strux.visualization.report_generator import HTMLReport

__version__ = "0.1.1"

__all__ = [
    "CSVDataSource",
    "DataSource",
    "Experiment",
    "HTMLReport",
    "Pipeline",
    "PostgresDataSource",
    "RegressionConfig",
    "RegressionResults",
    "Sequential",
    "ValidationLevel",
    "exact_match",
    "absolute_deviation",
    "relative_deviation",
    "subset",
]
