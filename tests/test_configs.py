"""Tests for regression configurations."""

import pytest
from enum import Enum
from typing import Any
from pydantic import BaseModel, ValidationError

from strux.configs import (
    RegressionConfig,
    ValidationLevel,
    FieldConfig
)
from strux.strategies import (
    ExactMatch,
    NumericRange,
    exact_match,
    numeric_range,
)

pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestCollectionWarning")

class SampleEnum(Enum):
    A = "a"
    B = "b"

class SampleSchema(BaseModel):
    enum_field: SampleEnum
    number_field: float
    optional_field: str | None = None

def test_regression_config_initialization() -> None:
    """Test RegressionConfig initialization with different field specifications."""
    config = RegressionConfig(
        SampleSchema,
        strict_fields=["enum_field"],
        relaxed_fields=["number_field"],
    )
    
    # Check field configurations
    assert "enum_field" in config.field_configs
    assert config.field_configs["enum_field"].level == ValidationLevel.STRICT
    assert isinstance(config.field_configs["enum_field"].strategy, ExactMatch)
    
    assert "number_field" in config.field_configs
    assert config.field_configs["number_field"].level == ValidationLevel.RELAXED
    assert isinstance(config.field_configs["number_field"].strategy, NumericRange)
    
    # Optional field should default to IGNORE
    assert config.field_configs["optional_field"].level == ValidationLevel.IGNORE

def test_field_config_validation() -> None:
    """Test field configuration validation."""
    # Valid configuration
    FieldConfig(
        strategy=exact_match(),
        threshold=None,
        level=ValidationLevel.STRICT,
    )
    
    # Invalid threshold for STRICT level
    with pytest.raises(ValidationError, match="Threshold is not allowed for STRICT level"):
        FieldConfig(
            strategy=exact_match(),
            threshold=0.8,  # Threshold not allowed for STRICT
            level=ValidationLevel.STRICT,
        )
    
    # Missing threshold for RELAXED level
    with pytest.raises(ValidationError, match="Threshold is required for RELAXED level"):
        FieldConfig(
            strategy=numeric_range(),
            threshold=None,  # Required for RELAXED
            level=ValidationLevel.RELAXED,
        )

def test_config_field_modification() -> None:
    """Test modifying field configurations."""
    config = RegressionConfig(SampleSchema)
    
    # Modify existing field
    config.configure_field(
        "enum_field",
        strategy=exact_match(),
        level=ValidationLevel.STRICT,
    )
    
    # Check modification
    field_config = config.field_configs["enum_field"]
    assert field_config.level == ValidationLevel.STRICT
    assert isinstance(field_config.strategy, ExactMatch)
    
    # Try to modify non-existent field
    with pytest.raises(ValueError, match="Unknown field"):
        config.configure_field(
            "non_existent_field",
            strategy=exact_match(),
        )
