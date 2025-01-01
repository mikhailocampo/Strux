"""Tests for regression testing results."""

import pytest
from datetime import datetime
from pydantic import BaseModel

from strux.results import FieldValidation, StepValidation, RegressionResults, ValidationStatus
from strux.configs import RegressionConfig, ValidationLevel


class SampleSchema(BaseModel):
    value: float
    text: str


def test_field_validation():
    """Test field validation creation and properties."""
    validation = FieldValidation(
        field_name="test_field",
        baseline_value=1.0,
        current_value=1.1,
        score=0.9,
        threshold=0.95,
        level=ValidationLevel.STRICT,
        status=ValidationStatus.FAILED,
        details={"diff": 0.1}
    )
    
    assert validation.field_name == "test_field"
    assert validation.status == ValidationStatus.FAILED
    assert validation.score == 0.9


def test_step_validation():
    """Test step validation aggregation."""
    field_validations = [
        FieldValidation(
            field_name="field1",
            baseline_value=1.0,
            current_value=1.0,
            score=1.0,
            threshold=0.95,
            level=ValidationLevel.STRICT,
            status=ValidationStatus.PASSED,
            details={}
        ),
        FieldValidation(
            field_name="field2",
            baseline_value=1.0,
            current_value=2.0,
            score=0.5,
            threshold=0.95,
            level=ValidationLevel.RELAXED,
            status=ValidationStatus.FAILED,
            details={}
        )
    ]
    
    step = StepValidation(
        step_name="test_step",
        field_validations=field_validations,
        metadata={"version": "1.0"}
    )
    
    assert not step.passed
    assert len(step.get_failed_validations()) == 1
    assert "field2" in step.format_summary()


def test_regression_results():
    """Test full regression results."""
    config = RegressionConfig(SampleSchema)
    step_validations = [
        StepValidation(
            step_name="step1",
            field_validations=[
                FieldValidation(
                    field_name="value",
                    baseline_value=1.0,
                    current_value=1.1,
                    score=0.9,
                    threshold=0.95,
                    level=ValidationLevel.STRICT,
                    status=ValidationStatus.FAILED,
                    details={}
                )
            ],
            metadata={}
        )
    ]
    
    results = RegressionResults(
        run_id="test_run",
        timestamp=datetime.now(),
        config=config,
        step_validations=step_validations
    )
    
    assert not results.passed
    assert len(results.get_failed_steps()) == 1
    assert "FAILED" in results.format_summary()
