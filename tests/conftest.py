"""Shared pytest fixtures for testing."""

from typing import Any

import pandas as pd
import pytest


@pytest.fixture
def sample_connection_params() -> dict[str, Any]:
    """Sample connection parameters for testing."""
    return {
        "host": "localhost",
        "port": 5432,
        "database": "test_db",
        "user": "test_user",
        "password": "test_password",
    }


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample DataFrame for testing."""
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
