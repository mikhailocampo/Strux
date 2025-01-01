"""Data loading utilities with built-in support for Postgres."""

import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol

import pandas as pd
from pydantic import BaseModel
import sqlalchemy


class SQLOperations(Protocol):
    """Protocol for SQL-specific operations."""

    def is_destructive_query(self, query: str) -> bool:
        """Check if the query contains destructive SQL operations."""
        ...


class DataSource(ABC):
    """Abstract base class for data sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the data source."""

    @property
    @abstractmethod
    def schema(self) -> type[BaseModel]:
        """Schema of the data source."""

    @abstractmethod
    def load_as_df(self) -> pd.DataFrame:
        """Load data as pandas DataFrame."""


# Error messages as constants
MISSING_PARAMS_ERROR = "Missing required connection parameters: {}"
DESTRUCTIVE_QUERY_ERROR = (
    "Destructive SQL operations (DROP, DELETE, etc.) are not allowed by default. "
    "Set allow_destructive=True to perform these operations."
)
DB_ERROR = "Error executing query: {}"


class PostgresDataSource(DataSource, SQLOperations):
    """PostgreSQL data source for loading data to pandas DataFrame.

    Example:
        >>> source = PostgresDataSource(
                connection_params={
                    "host": "localhost",
                    "port": 5432,
                    "database": "mydatabase",
                    "user": "myuser",
                    "password": "mypassword"
                },
                query="SELECT * FROM mytable"
            )
        >>> df = source.load_as_df()

    """

    DESTRUCTIVE_KEYWORDS: ClassVar[set[str]] = {
        "DROP",
        "DELETE",
        "TRUNCATE",
        "ALTER",
        "UPDATE",
        "CREATE",
        "REPLACE",
        "INSERT",
        "GRANT",
        "REVOKE",
    }

    def __init__(
        self,
        connection_params: dict[str, Any],
        query: str,
        input_schema: type[BaseModel] = None,
        *,  # Force keyword arguments for boolean flags
        allow_destructive: bool = False,
    ) -> None:
        """Initialize PostgreSQL data source.

        Args:
            connection_params: Dictionary of connection parameters:
                - host: str, the host name of the database server
                - port: int, the port number of the database server
                - database: str, the name of the database
                - user: str, the username for the database
                - password: str, the password for the database
            query: SQL query to execute
            input_schema: Optional schema to use for the query
            allow_destructive: Whether to allow destructive operations

        """
        self._connection_params = connection_params
        self._query = query
        self._allow_destructive = allow_destructive
        self._validate_connection_params()
        self._name = f"postgres_{connection_params.get('database', 'unknown')}"
        self._input_schema = input_schema
    
    @property
    def name(self) -> str:
        """Name of the data source."""
        return self._name
    
    @property
    def schema(self) -> type[BaseModel]:
        """Schema of the data source."""
        return self._input_schema

    @property
    def connection_params(self) -> dict[str, Any]:
        """Connection parameters for the data source."""
        return self._connection_params

    @property
    def query(self) -> str:
        """SQL query to execute."""
        return self._query

    @property
    def allow_destructive(self) -> bool:
        """Whether to allow destructive operations."""
        return self._allow_destructive

    def _is_destructive_query(self, query: str) -> bool:
        """Check if the query contains destructive SQL operations.

        Args:
            query: SQL query to check

        Returns:
            bool: True if query contains destructive operations

        """
        # Remove SQL comments and normalize whitespace
        clean_query = re.sub(r"--.*$|\s+", " ", query, flags=re.MULTILINE).strip()

        # Extract the first word (usually the SQL command)
        first_word = clean_query.split(" ")[0].upper()

        return first_word in self.DESTRUCTIVE_KEYWORDS

    def _validate_connection_params(self) -> None:
        """Validate that required connection parameters are present."""
        required_params = {"host", "database", "user", "password"}
        missing_params = required_params - set(self.connection_params.keys())

        if missing_params:
            raise ValueError(MISSING_PARAMS_ERROR.format(", ".join(missing_params)))

    def _build_connection_string(self) -> str:
        """Build SQLAlchemy connection string from connection parameters."""
        params = self.connection_params.copy()
        port = params.pop("port", 5432)

        return (
            f"postgresql://{params['user']}:{params['password']}"
            f"@{params['host']}:{port}/{params['database']}"
        )

    def load_as_df(self) -> pd.DataFrame:
        """Execute query and return results as a pandas DataFrame.

        Returns:
            pandas.DataFrame: The results of the query as a DataFrame.

        Raises:
            ValueError: If destructive query is attempted without permission.
            RuntimeError: If there is an error executing the query.

        """
        if not self.allow_destructive and self._is_destructive_query(self.query):
            raise ValueError(DESTRUCTIVE_QUERY_ERROR)

        try:
            engine = sqlalchemy.create_engine(self._build_connection_string())
            with engine.connect() as connection:
                return pd.read_sql(self.query, connection)
        except sqlalchemy.exc.SQLAlchemyError as e:
            raise RuntimeError(DB_ERROR.format(str(e))) from e
