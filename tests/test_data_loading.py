"""Tests for data loading functionality."""

from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from pytest_mock import MockerFixture
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel

from strux.data_loading import DataSource, PostgresDataSource


class SampleSchema(BaseModel):
    value: float
    text: str


def test_postgres_data_source_initialization(
    sample_connection_params: dict[str, Any],
) -> None:
    """Test PostgresDataSource initialization and validation."""
    # Missing required parameters
    with pytest.raises(
        ValueError, match="Missing required connection parameters"
    ) as exc_info:
        PostgresDataSource(
            connection_params={}, 
            query="SELECT 1",
            input_schema=SampleSchema
        )
    assert "Missing required connection parameters" in str(exc_info.value)

    # All required params present
    source = PostgresDataSource(
        connection_params=sample_connection_params,
        query="SELECT 1",
        input_schema=SampleSchema,
    )
    assert isinstance(source, DataSource)
    assert source.schema == SampleSchema  # Verify schema is set correctly


def test_postgres_connection_string_building(
    sample_connection_params: dict[str, Any],
) -> None:
    """Test the connection strings are built correctly."""
    source = PostgresDataSource(
        connection_params=sample_connection_params,
        query="SELECT 1",
        input_schema=SampleSchema,
    )

    conn_str = source._build_connection_string()
    assert "postgresql://" in conn_str
    assert "localhost" in conn_str
    assert "5432" in conn_str
    assert "test_db" in conn_str
    assert "test_user" in conn_str
    assert "test_password" in conn_str


def test_destructive_query_detection(
    sample_connection_params: dict[str, Any],
) -> None:
    """Test that destructive queries are detected correctly."""
    source = PostgresDataSource(
        connection_params=sample_connection_params,
        query="SELECT 1",
        input_schema=SampleSchema,
    )

    # Test various SQL commands
    assert not source._is_destructive_query("SELECT * FROM table")
    assert source._is_destructive_query("DROP TABLE IF EXISTS test_table")
    assert source._is_destructive_query("DELETE FROM test_table")
    assert source._is_destructive_query("CREATE TABLE test_table (id INT)")

    # Test with comments and whitespace
    assert source._is_destructive_query("""
        -- This is a comment
        DROP TABLE IF EXISTS test_table
    """)

    # Test case insensitive matching
    assert source._is_destructive_query("drop table if exists test_table")


def test_destructive_query_prevention(
    sample_connection_params: dict[str, Any],
    mock_engine: tuple[Any, Any],
) -> None:
    """Test that destructive queries are prevented by default."""
    source = PostgresDataSource(
        connection_params=sample_connection_params,
        query="DROP TABLE IF EXISTS test_table",
        input_schema=SampleSchema,
    )

    # Should raise ValueError by default
    with pytest.raises(ValueError, match="Destructive SQL operations"):
        source.load_as_df()

    # Should allow when explicitly set
    source = PostgresDataSource(
        connection_params=sample_connection_params,
        query="DROP TABLE IF EXISTS test_table",
        input_schema=SampleSchema,
        allow_destructive=True,
    )

    # Mock pandas.read_sql
    with (
        patch("pandas.read_sql", side_effect=SQLAlchemyError("Test database error")),
        pytest.raises(RuntimeError),
    ):
        source.load_as_df()


@pytest.fixture
def mock_engine(mocker: MockerFixture) -> tuple[Any, Any]:
    """Fixture to mock SQLAlchemy engine and connection."""
    mock_engine = mocker.patch("sqlalchemy.create_engine")
    mock_connection = mocker.MagicMock()
    mock_engine.return_value.connect.return_value.__enter__.return_value = mock_connection
    return mock_engine, mock_connection


def test_successful_query_execution(
    mock_engine: tuple[Any, Any],
    sample_connection_params: dict[str, Any],
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test successful query execution with mocked database connection."""
    source = PostgresDataSource(
        connection_params=sample_connection_params,
        query="SELECT * FROM mytable",
        input_schema=SampleSchema,
    )

    with patch("pandas.read_sql", return_value=sample_dataframe):
        result = source.load_as_df()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "col1" in result.columns
    assert "col2" in result.columns


def test_database_error_handling(
    mock_engine: tuple[Any, Any],
    sample_connection_params: dict[str, Any],
) -> None:
    """Test handling of database errors."""
    source = PostgresDataSource(
        connection_params=sample_connection_params,
        query="SELECT * FROM mytable",
        input_schema=SampleSchema,
    )

    with (
        patch("pandas.read_sql", side_effect=SQLAlchemyError("Test database error")),
        pytest.raises(RuntimeError),
    ):
        source.load_as_df()
