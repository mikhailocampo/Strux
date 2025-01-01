"""Built-in comparison strategies for fields."""

from enum import Enum
from typing import Any, TypeVar, get_origin, Union, Sequence

from strux.configs import ComparisonStrategy

T = TypeVar("T")
Number = Union[int, float]
Collection = Union[list, set, Sequence]


class ExactMatch(ComparisonStrategy):
    """Strategy requiring exact matches.

    Use this for fields where values must match exactly like enums,
    identifiers, or categorical values.

    Example:
        >>> config.configure_field("status", strategy=exact_match())
    """

    def compare(self, baseline: Any, current: Any) -> float:
        """Compare values for exact equality.

        Returns:
            1.0 if values are exactly equal, 0.0 otherwise.
        """
        return 1.0 if baseline == current else 0.0

    def validate(self, field_type: type) -> bool:
        """Can be used with any type."""
        return True


class NumericRange(ComparisonStrategy):
    """Strategy for comparing numeric values within a tolerance range.

    Args:
        tolerance: Maximum allowed deviation as a fraction (0.1 = 10%)

    Example:
        >>> config.configure_field(
                "price",
                strategy=numeric_range(tolerance=0.1)
            )
    """

    def __init__(self, tolerance: float = 0.1) -> None:
        if not 0 < tolerance <= 1:
            raise ValueError("Tolerance must be between 0 and 1")
        self.tolerance = tolerance

    def compare(self, baseline: Number | None, current: Number) -> float:
        """Compare numberic values within a tolerance range.

        Returns:
            Score between 0.0 and 1.0 based on how close the values are.
        """
        if baseline is None:
            return 1.0 
        if baseline == 0:
            return 1.0 if current == 0 else 0.0
        diff = abs(baseline - current) / abs(baseline)
        return max(0.0, 1.0 - (diff / self.tolerance))

    def validate(self, field_type: type) -> bool:
        """Can be used with int or float."""
        return field_type in (int, float)


class SubsetMatch(ComparisonStrategy):
    """Strategy for comparing lists/sets allowing partial matches.

    Useful for fields like tags or categories where partial matches are acceptable.

    Example:
        >>> config.configure_field(
                "tags",
                strategy=subset(threshold=0.8),
            )
    """
    
    def __init__(self, threshold: float = 0.8) -> None:
        """Initialize with minimum required overlap ratio."""
        self.threshold = threshold

    def compare(self, baseline: Collection, current: Collection) -> float:
        """Compare collections using Jaccard similarity.

        Jaccard similarity is a measure of similarity between two sets.
        It is the ratio of the size of the intersection to the size of the union.

        Returns:
            Score between 0.0 and 1.0 based on how similar the collections are.
        """
        baseline_set = set(baseline)
        current_set = set(current)

        if not baseline_set:
            return 1.0 if not current_set else 0.0

        intersection = baseline_set.intersection(current_set)
        return len(intersection) / len(baseline_set)

    def validate(self, field_type: type) -> bool:
        """Can be used with collection types."""
        # Handle both raw types and parameterized types
        if field_type in (list, set):
            return True
        
        # Check if the type is a generic type with origin list, set, or Sequence
        origin = get_origin(field_type)
        return origin in (list, set, Sequence)


# Convenience functions for creating strategies
def exact_match() -> ExactMatch:
    """Create an exact match strategy.

    Use for fields requiring exact equality.
    """
    return ExactMatch()


def numeric_range(tolerance: float = 0.1) -> NumericRange:
    """Create a numeric range strategy.

    Args:
        tolerance: Maximum allowed deviation as a fraction (0.1 = 10%)
    """
    return NumericRange(tolerance=tolerance)


def subset(threshold: float = 0.8) -> SubsetMatch:
    """Create a subset match strategy.

    Args:
        threshold: Minimum required overlap ratio
    """
    return SubsetMatch(threshold=threshold)
