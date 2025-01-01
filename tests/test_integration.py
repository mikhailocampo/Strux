"""Integration tests for the full pipeline."""

from typing import Any
import pandas as pd
from pydantic import BaseModel

from strux.configs import RegressionConfig, ValidationLevel
from strux.data_loading import DataSource
from strux.pipeline import Sequential


# Test models
class InputSchema(BaseModel):
    value: float
    text: str

class OutputSchema(BaseModel):
    value: float
    text: str
    processed: bool

# Mock data source
class MockDataSource(DataSource):
    def __init__(self) -> None:
        self._name = "mock_source"
        self._schema = InputSchema

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> type[BaseModel]:
        return self._schema

    def load_as_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"value": 1.0, "text": "test1"},
            {"value": 2.0, "text": "test2"},
        ])

# Mock inference function
def mock_inference(data: InputSchema) -> OutputSchema:
    return OutputSchema(
        value=data.value,
        text=data.text.upper(),
        processed=True
    )

def test_full_pipeline():
    """Test the full pipeline execution."""
    # Setup
    data_source = MockDataSource()
    config = RegressionConfig(
        OutputSchema,
        strict_fields=["processed"],
        relaxed_fields=["value"],
        # Text field will be ignored by default
    )

    # Create pipeline
    pipeline = Sequential.from_steps(
        data_source=data_source,
        steps=[
            ("process", mock_inference, OutputSchema),
        ],
        config=config,
    )

    # Run pipeline
    results = pipeline.run()

    # Verify results
    assert results.run_id.startswith("mock_source_")
    assert len(results.step_validations) == 2  # One for each input row
    
    # Check first row results
    step = results.step_validations[0]
    assert step.step_name == "process"
    assert len(step.field_validations) == 3  # All fields are validated
    
    # Verify field validations
    value_validation = next(v for v in step.field_validations if v.field_name == "value")
    assert value_validation.current_value == 1.0
    assert value_validation.level == ValidationLevel.RELAXED
    
    processed_validation = next(v for v in step.field_validations if v.field_name == "processed")
    assert processed_validation.current_value is True
    assert processed_validation.level == ValidationLevel.STRICT

    text_validation = next(v for v in step.field_validations if v.field_name == "text")
    assert text_validation.level == ValidationLevel.IGNORE  # Default level
