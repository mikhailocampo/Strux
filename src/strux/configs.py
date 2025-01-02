"""Configuration classes for regression testing."""

from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class ValidationLevel(Enum):
    """Level of validation strictness."""

    STRICT = "strict"
    RELAXED = "relaxed"
    WARNING = "warning"


class FieldConfig(BaseModel):
    """Configuration for field validation."""

    threshold: float | None = None
    level: ValidationLevel = ValidationLevel.STRICT
    strategy: Any = None  # ComparisonStrategy instance


class RegressionConfig(BaseModel, Generic[T]):
    """Configuration for regression testing."""

    target_schema: type[T]
    field_configs: dict[str, FieldConfig] = Field(default_factory=dict)

    def configure_field(
        self,
        field_name: str,
        *,
        threshold: float | None = None,
        level: ValidationLevel = ValidationLevel.STRICT,
        strategy: Any = None,
    ) -> None:
        """Configure validation for a specific field."""
        self.field_configs[field_name] = FieldConfig(
            threshold=threshold,
            level=level,
            strategy=strategy,
        )
