"""Configuration classes for regression testing."""

from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar, get_origin

from pydantic import BaseModel, ConfigDict


class ValidationLevel(Enum):
    """Level of validation strictness."""

    STRICT = "strict"  # Must match exactly
    RELAXED = "relaxed"  # Allow some deviations
    IGNORE = "ignore"  # Ignore all validation errors


class ComparisonStrategy(ABC):
    """Protocol defining how to compare field values."""

    @abstractmethod
    def compare(self, baseline: Any, current: Any) -> float:
        """Compare two values and return a score."""
        pass

    @abstractmethod
    def validate(self, field_type: type) -> bool:
        """Validate if this strategy can be applied to the given field type."""
        pass


class FieldConfig(BaseModel):
    """Configuration for how to compare a specific field."""

    strategy: ComparisonStrategy
    threshold: float | None = None
    level: ValidationLevel = ValidationLevel.STRICT
    description: str | None = None
    pre_compare: Callable | None = None
    post_compare: Callable | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def model_post_init(self, _: Any) -> None:
        """Validate configuration after initialization."""
        if self.level == ValidationLevel.STRICT and self.threshold is not None:
            raise ValueError("Threshold is not allowed for STRICT level")
        if self.level == ValidationLevel.RELAXED and self.threshold is None:
            raise ValueError("Threshold is required for RELAXED level")


T = TypeVar("T", bound=BaseModel)


class RegressionConfig(Generic[T]):
    """Configuration for regression testing with smart defaults."""

    def __init__(
        self,
        target_schema: type[T],
        *,
        strict_fields: list[str] | None = None,
        relaxed_fields: list[str] | None = None,
        ignore_fields: list[str] | None = None,
    ) -> None:
        """Initialize regression configuration.

        Args:
            target_schema: The schema of the target model to compare against.
            strict_fields: List of fields to validate strictly.
            relaxed_fields: List of fields to validate with a relaxed threshold.
            ignore_fields: List of fields to ignore completely.
        """
        self.target_schema = target_schema
        self.field_configs: dict[str, FieldConfig] = {}

        # Configure fields based on input lists
        all_fields = target_schema.model_fields

        for field_name, field in all_fields.items():
            # Default to IGNORE if not specified
            if ignore_fields and field_name in ignore_fields:
                level = ValidationLevel.IGNORE
            elif strict_fields and field_name in strict_fields:
                level = ValidationLevel.STRICT
            elif relaxed_fields and field_name in relaxed_fields:
                level = ValidationLevel.RELAXED
            else:
                level = ValidationLevel.IGNORE

            strategy = self._get_default_strategy(field)
            self.field_configs[field_name] = FieldConfig(
                strategy=strategy,
                level=level,
                threshold=0.9 if level == ValidationLevel.RELAXED else None,
            )

    def _get_default_strategy(self, field: Any) -> ComparisonStrategy:
        """Get default comparison strategy based on field type."""
        from strux.strategies import exact_match, numeric_range, subset

        field_type = field.annotation
        origin = get_origin(field_type)

        if origin is None:
            if issubclass(field_type, Enum):
                return exact_match()
            elif field_type in (int, float):
                return numeric_range()
            elif field_type == str:
                return exact_match()
        elif origin in (list, set):
            return subset()

        return exact_match()  # Default to exact match for complex types

    def configure_field(
        self,
        field_name: str,
        strategy: ComparisonStrategy | None = None,
        level: ValidationLevel | None = None,
        threshold: float | None = None,
    ) -> "RegressionConfig[T]":
        """Configure a specific field.

        Args:
            field_name: Name of field to configure
            strategy: Optional comparison strategy
            level: Optional validation level
            threshold: Optional comparison threshold

        Returns:
            self for method chaining
        """
        if field_name not in self.field_configs:
            raise ValueError(f"Unknown field: {field_name}")

        config = self.field_configs[field_name]

        if strategy:
            config.strategy = strategy
        if level:
            config.level = level
        if threshold is not None:
            config.threshold = threshold

        return self

    @classmethod
    def from_annotations(cls, source_path: str) -> "RegressionConfig[T]":
        """Create a config from a source file annotations."""
        # TODO: Future enhancement
        raise NotImplementedError
