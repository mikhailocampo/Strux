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
    strategy: Any = None
    compare_with_annotation: bool = False


class RegressionConfig(BaseModel, Generic[T]):
    """Configuration for regression testing."""
    target_schema: type[T]
    field_configs: dict[str, FieldConfig] = Field(default_factory=dict)
    annotation_field: str | None = None
    data_source: Any = None

    def configure_field(
        self,
        field_name: str,
        *,
        threshold: float | None = None,
        level: ValidationLevel = ValidationLevel.STRICT,
        strategy: Any = None,
        compare_with_annotation: bool = False,
    ) -> "RegressionConfig[T]":
        """Configure validation for a specific field."""
        if compare_with_annotation and not self.annotation_field:
            raise ValueError("Must set annotation_field in config to use compare_with_annotation")
            
        self.field_configs[field_name] = FieldConfig(
            threshold=threshold,
            level=level,
            strategy=strategy,
            compare_with_annotation=compare_with_annotation,
        )
        return self
