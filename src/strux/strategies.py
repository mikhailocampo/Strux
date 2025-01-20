"""Built-in comparison strategies for fields."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar, get_origin, Any, List, Optional, Tuple, Dict

T = TypeVar("T")
Number = int | float
Collection = list | set | Sequence

# Error messages
TOLERANCE_ERROR_MSG = "Tolerance must be between 0 and 1"
UNSUPPORTED_TYPE_MSG = "Unsupported type: {}"
EXPECTED_LIST_MSG = "Expected list but got {}"


class ComparisonStrategy(ABC):
    """Base class for comparison strategies.

    Defines the interface for comparing values in regression testing.
    """

    @abstractmethod
    def compare(self, baseline: Number | Collection, current: Number | Collection) -> float:
        """Compare two values as a float."""

    @abstractmethod
    def validate(self, field_type: type) -> bool:
        """Validate the strategy against a field type."""


class ValidationStrategy(ABC):
    """Base class for validation strategies."""
    
    @abstractmethod
    def validate(self, predictions: List[Any], annotations: Optional[List[Any]] = None) -> float:
        """Validate predictions against annotations.
        
        Args:
            predictions: List of predicted values
            annotations: Optional list of annotation values
            
        Returns:
            float: Score between 0 and 1
        """
        pass


class ExactMatch(ValidationStrategy):
    """Strategy for exact matching of values."""
    
    def validate(self, predictions: List[Any], annotations: Optional[List[Any]] = None) -> float:
        """Validate exact matches between predictions and annotations.
        
        Args:
            predictions: List of predicted values
            annotations: List of expected values
            
        Returns:
            float: Percentage of exact matches (0-1)
        """
        if not annotations:
            return 0.0
            
        matches = sum(1 for p, a in zip(predictions, annotations) if p == a)
        return matches / len(predictions)


class AbsoluteDeviation(ValidationStrategy):
    """Strategy for comparing values within a fixed ±threshold.
    
    Example:
        >>> config.configure_field(
                "price",
                strategy=absolute_deviation(50000),  # Allow ±$50k difference
                threshold=0.9  # 90% must be within threshold
            )
    """
    
    def __init__(self, threshold: float):
        self.threshold = threshold
    
    def validate(self, predictions: List[float], annotations: Optional[List[float]] = None) -> float:
        """Validate numeric predictions are within threshold of annotations.
        
        Args:
            predictions: List of predicted values
            annotations: List of expected values
            
        Returns:
            float: Percentage within threshold (0-1)
        """
        if not annotations:
            return 0.0
            
        within_threshold = sum(
            1 for p, a in zip(predictions, annotations)
            if abs(p - a) <= self.threshold
        )
        return within_threshold / len(predictions)


class RelativeDeviation(ComparisonStrategy):
    """Strategy for comparing values within a relative percentage difference.
    
    Example:
        >>> config.configure_field(
                "price",
                strategy=relative_deviation(0.1),  # Allow ±10% difference
                threshold=0.9
            )
    """

    def __init__(self, percentage: float):
        self.percentage = percentage

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
        return max(0.0, 1.0 - (diff / self.percentage))

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
        # Handle None type
        if field_type is None:
            return False

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


def absolute_deviation(threshold: float) -> AbsoluteDeviation:
    """Create strategy that checks if values are within fixed ±threshold."""
    return AbsoluteDeviation(threshold)


def relative_deviation(percentage: float) -> RelativeDeviation:
    """Create strategy that checks if values are within ±percentage difference."""
    return RelativeDeviation(percentage)


def subset(threshold: float = 0.8) -> SubsetMatch:
    """Create a subset match strategy.

    Args:
        threshold: Minimum required overlap ratio

    """
    return SubsetMatch(threshold=threshold)
