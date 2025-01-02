"""Data loading utilities with built-in support for Postgres."""

import json
import re
from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar, get_args, get_origin

import pandas as pd
import sqlalchemy
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Error messages as constants
MISSING_PARAMS_ERROR = "Missing required connection parameters: {}"
DESTRUCTIVE_QUERY_ERROR = (
    "Destructive SQL operations (DROP, DELETE, etc.) are not allowed by default. "
    "Set allow_destructive=True to perform these operations."
)
DB_ERROR = "Error executing query: {}"


class DataSource(Generic[T]):
    """Base class for data sources with smart JSON handling."""

    @classmethod
    def from_postgres(
        cls,
        schema: type[T],
        query: str,
        connection_params: dict,
        *,
        transform: dict[str, Callable] | None = None,
        json_columns: dict[str, type] | None = None,
    ) -> "PostgresDataSource[T]":
        """Create a data source from Postgres with smart JSON handling.

        Args:
            schema: Pydantic model defining the expected data structure
            query: SQL query to execute
            connection_params: Database connection parameters
            transform: Optional field-specific transformations
            json_columns: Mapping of column names to their expected types for JSON/JSONB columns

        """
        return PostgresDataSource(
            schema=schema,
            query=query,
            connection_params=connection_params,
            field_transformers=transform,
            json_columns=json_columns,
        )


class PostgresDataSource(DataSource[T]):
    """PostgreSQL data source with JSON support and schema validation.

    This class is a wrapper around the SQLAlchemy library for loading data from a PostgreSQL database.

    It supports automatic JSON column detection, field transformations, and schema validation.

    Example usage:

    ```python
    from strux.data_loading import PostgresDataSource
    from pydantic import BaseModel

    class MyModel(BaseModel):
        id: int
        name: str

    data_source = PostgresDataSource.from_postgres(
        schema=MyModel,
        query="SELECT id, name FROM my_table",
        connection_params={"host": "localhost", "database": "my_db", "user": "my_user", "password": "my_password"},
    )
    ```

    This will load data from the `my_table` table into a pandas DataFrame and validate it against the `MyModel` schema.
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
        schema: type[T],
        query: str,
        connection_params: dict,
        *,
        field_transformers: dict[str, Callable] | None = None,
        json_columns: dict[str, type] | None = None,
        allow_destructive: bool = False,
    ):
        """Initialize PostgreSQL data source.

        Args:
            schema: The schema of the expected data
            query: SQL query to execute
            connection_params: Database connection parameters
            field_transformers: Optional field-specific transformations
            json_columns: Mapping of column names to their JSON types
            allow_destructive: Whether to allow destructive operations

        """
        self.schema = schema
        self.query = query
        self.connection_params = connection_params
        self.field_transformers = field_transformers or {}
        self.json_columns = json_columns or {}
        self.allow_destructive = allow_destructive

        # Auto-detect JSON fields if not specified
        if not json_columns:
            self._auto_detect_json_fields()

        self._validate_connection_params()

    def _auto_detect_json_fields(self) -> None:
        """Automatically detect fields that might be JSON based on schema types."""
        for field_name, field in self.schema.model_fields.items():
            if field_name not in self.json_columns:
                # Check if field type suggests JSON (List, Dict, or BaseModel)
                origin = get_origin(field.annotation)
                if origin in (list, dict) or issubclass(get_args(field.annotation)[0], BaseModel):
                    self.json_columns[field_name] = field.annotation

    def _transform_json_column(self, data: Any, target_type: type) -> Any:
        """Transform JSON data into the target type."""
        if data is None:
            return None

        # If data is already a dict/list, use it as is, otherwise parse JSON
        if isinstance(data, (dict, list)):
            json_data = data
        else:
            try:
                json_data = json.loads(data)
            except (TypeError, json.JSONDecodeError) as e:
                raise ValueError(f"Failed to parse JSON data: {e!s}\nData: {data}")

        # Handle different target types
        origin = get_origin(target_type)
        if origin is list:
            # Handle List[SomeType]
            item_type = get_args(target_type)[0]
            if not isinstance(json_data, list):
                raise ValueError(f"Expected list but got {type(json_data)}")

            return [self._transform_single_value(item, item_type) for item in json_data]
        if origin is None:
            # Handle non-generic types (dict, BaseModel, etc)
            return self._transform_single_value(json_data, target_type)
        raise ValueError(f"Unsupported type: {target_type}")

    def _transform_single_value(self, data: Any, target_type: type) -> dict | BaseModel:
        """Transform a single value into the target type."""
        if target_type == dict:
            return data
        if issubclass(target_type, BaseModel):
            return target_type.model_validate(data)
        return data

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

        host = params["host"]
        if ".supabase.co" in host:
            return (
                f"postgresql://{params['user']}:{params['password']}"
                f"@{host}:{port}/{params['database']}"
                "?sslmode=require"
            )

        return f"postgresql://{params['user']}:{params['password']}" f"@{host}:{port}/{params['database']}"

    def _is_destructive_query(self, query: str) -> bool:
        """Check if the query contains destructive SQL operations."""
        clean_query = re.sub(r"--.*$|\s+", " ", query, flags=re.MULTILINE).strip()
        first_word = clean_query.split(" ")[0].upper()
        return first_word in self.DESTRUCTIVE_KEYWORDS

    def load_as_df(self) -> pd.DataFrame:
        """Load and transform data, handling JSON columns automatically."""
        if not self.allow_destructive and self._is_destructive_query(self.query):
            raise ValueError(DESTRUCTIVE_QUERY_ERROR)

        try:
            engine = sqlalchemy.create_engine(self._build_connection_string())
            with engine.connect() as connection:
                df = pd.read_sql(self.query, connection)

                if df.empty:
                    return df

                # Handle JSON columns first
                for col_name, target_type in self.json_columns.items():
                    if col_name in df.columns:
                        df[col_name] = df[col_name].apply(lambda x: self._transform_json_column(x, target_type))

                # Then handle any custom transformations
                transformed_data = {}
                for field_name, field_info in self.schema.model_fields.items():
                    if field_name in df.columns:
                        transformed_data[field_name] = df[field_name]
                    elif field_name in self.field_transformers:
                        transform_fn = self.field_transformers[field_name]
                        transformed_data[field_name] = transform_fn(df)
                    else:
                        raise ValueError(
                            f"No mapping found for field '{field_name}'. " f"Available columns: {list(df.columns)}"
                        )

                return pd.DataFrame(transformed_data)

        except sqlalchemy.exc.SQLAlchemyError as e:
            raise RuntimeError(DB_ERROR.format(str(e))) from e
