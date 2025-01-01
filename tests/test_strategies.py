"""Tests for comparison strategies."""

import pytest
from enum import Enum
from typing import Any

from strux.strategies import (
    ExactMatch,
    NumericRange,
    SubsetMatch,
    exact_match,
    numeric_range,
    subset,
)


class SampleEnum(Enum):
    A = "a"
    B = "b"


@pytest.fixture
def exact_strategy():
    """Fixture for ExactMatch strategy."""
    return exact_match()


@pytest.fixture
def numeric_strategy():
    """Fixture for NumericRange strategy."""
    return numeric_range(tolerance=0.1)


@pytest.fixture
def subset_strategy():
    """Fixture for SubsetMatch strategy."""
    return subset()


class TestExactMatch:
    """Tests for ExactMatch strategy."""
    
    def test_basic_equality(self, exact_strategy):
        """Test basic equality comparisons."""
        assert exact_strategy.compare("hello", "hello") == 1.0
        assert exact_strategy.compare("hello", "world") == 0.0
    
    def test_different_types(self, exact_strategy):
        """Test comparisons with different types."""
        assert exact_strategy.compare(123, 123) == 1.0
        assert exact_strategy.compare(1.0, 1.0) == 1.0
        assert exact_strategy.compare(True, True) == 1.0
    
    def test_enums(self, exact_strategy):
        """Test enum comparisons."""
        assert exact_strategy.compare(SampleEnum.A, SampleEnum.A) == 1.0
        assert exact_strategy.compare(SampleEnum.A, SampleEnum.B) == 0.0
    
    def test_collections(self, exact_strategy):
        """Test collection comparisons."""
        assert exact_strategy.compare([1, 2, 3], [1, 2, 3]) == 1.0
        assert exact_strategy.compare([1, 2, 3], [3, 2, 1]) == 0.0
    
    def test_validation(self, exact_strategy):
        """Test type validation."""
        assert exact_strategy.validate(str)
        assert exact_strategy.validate(int)
        assert exact_strategy.validate(SampleEnum)


class TestNumericRange:
    """Tests for NumericRange strategy."""
    
    def test_exact_matches(self, numeric_strategy):
        """Test exact numeric matches."""
        assert numeric_strategy.compare(100, 100) == 1.0
        assert numeric_strategy.compare(0, 0) == 1.0
    
    def test_within_tolerance(self, numeric_strategy):
        """Test values within tolerance."""
        assert numeric_strategy.compare(100, 105) == 0.5  # 5% difference
        assert numeric_strategy.compare(100, 110) == 0.0  # 10% difference
    
    def test_outside_tolerance(self, numeric_strategy):
        """Test values outside tolerance."""
        assert numeric_strategy.compare(100, 115) == 0.0  # 15% difference
    
    def test_validation(self, numeric_strategy):
        """Test type validation."""
        assert numeric_strategy.validate(int)
        assert numeric_strategy.validate(float)
        assert not numeric_strategy.validate(str)
    
    def test_invalid_initialization(self):
        """Test invalid tolerance values."""
        with pytest.raises(ValueError):
            numeric_range(tolerance=-0.1)
        with pytest.raises(ValueError):
            numeric_range(tolerance=1.1)


class TestSubsetMatch:
    """Tests for SubsetMatch strategy."""
    
    def test_exact_matches(self, subset_strategy):
        """Test exact collection matches."""
        assert subset_strategy.compare([1, 2, 3], [1, 2, 3]) == 1.0
        assert subset_strategy.compare(set([1, 2]), set([1, 2])) == 1.0
    
    def test_partial_matches(self, subset_strategy):
        """Test partial collection matches."""
        assert subset_strategy.compare([1, 2, 3], [1, 2]) == 2/3
        assert subset_strategy.compare(set([1, 2, 3]), set([1, 3])) == 2/3
    
    def test_no_matches(self, subset_strategy):
        """Test collections with no matches."""
        assert subset_strategy.compare([1, 2, 3], [4, 5, 6]) == 0.0
    
    def test_empty_collections(self, subset_strategy):
        """Test empty collection handling."""
        assert subset_strategy.compare([], []) == 1.0
        assert subset_strategy.compare([1, 2], []) == 0.0
    
    def test_validation(self, subset_strategy):
        """Test type validation."""
        assert subset_strategy.validate(list)
        assert subset_strategy.validate(set)
        assert not subset_strategy.validate(str)
