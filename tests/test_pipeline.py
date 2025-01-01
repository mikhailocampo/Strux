"""Tests for pipeline execution."""

import pytest
import pandas as pd
from datetime import datetime
from pydantic import BaseModel
from typing import Any

from strux.pipeline import Pipeline, Sequential
from strux.configs import RegressionConfig
from strux.data_loading import DataSource
from strux.step import Step


class MockDataSource(DataSource):
    """Mock data source for testing."""
    
    def __init__(self, data: list[dict]) -> None:
        self._data = data
        self._name = "mock_source"
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def schema(self) -> type[BaseModel]:
        return InputSchema
    
    def load_as_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._data)


class InputSchema(BaseModel):
    text: str
    value: int


class IntermediateSchema(BaseModel):
    processed: str
    score: float


class OutputSchema(BaseModel):
    final: str
    confidence: float


def test_pipeline_initialization():
    """Test basic pipeline setup."""
    pipeline = Sequential[OutputSchema](
        data_source=MockDataSource([{"text": "test", "value": 1}]),
        config=RegressionConfig(OutputSchema)
    )
    
    assert not pipeline._built
    assert len(pipeline._steps) == 0


def test_pipeline_step_addition():
    """Test adding steps to pipeline."""
    def step1(data: InputSchema) -> IntermediateSchema:
        return IntermediateSchema(
            processed=data.text.upper(),
            score=float(data.value)
        )
    
    def step2(data: IntermediateSchema) -> OutputSchema:
        return OutputSchema(
            final=data.processed,
            confidence=data.score * 100
        )
    
    pipeline = Sequential[OutputSchema](
        data_source=MockDataSource([{"text": "test", "value": 1}]),
        config=RegressionConfig(OutputSchema)
    ) \
    .add_step(
        inference_fn=step1,
        input_schema=InputSchema,
        output_schema=IntermediateSchema
    ) \
    .add_step(
        inference_fn=step2,
        input_schema=IntermediateSchema,
        output_schema=OutputSchema
    ) \
    .build()
    
    assert len(pipeline._steps) == 2
    assert pipeline._built


def test_pipeline_execution():
    """Test running full pipeline."""
    data = [{"text": "hello", "value": 1}]
    
    def step1(data: InputSchema) -> IntermediateSchema:
        return IntermediateSchema(
            processed=data.text.upper(),
            score=float(data.value)
        )
    
    def step2(data: IntermediateSchema) -> OutputSchema:
        return OutputSchema(
            final=data.processed,
            confidence=data.score * 100
        )
    
    pipeline = Sequential[OutputSchema](
        data_source=MockDataSource(data),
        config=RegressionConfig(OutputSchema)
    ) \
    .add_step(
        inference_fn=step1,
        input_schema=InputSchema,
        output_schema=IntermediateSchema
    ) \
    .add_step(
        inference_fn=step2,
        input_schema=IntermediateSchema,
        output_schema=OutputSchema
    ) \
    .build()
    
    results = pipeline.run()
    
    assert results.run_id is not None
    assert isinstance(results.timestamp, datetime)
    assert len(results.step_validations) == 2


def test_unbuilt_pipeline_execution():
    """Test running pipeline without steps."""
    pipeline = Sequential[OutputSchema](
        data_source=MockDataSource([{"text": "test", "value": 1}]),
        config=RegressionConfig(OutputSchema)
    )
    
    with pytest.raises(RuntimeError, match="Pipeline must be built before running"):
        pipeline.run()


def test_incompatible_step_schemas():
    """Test adding steps with incompatible schemas."""
    pipeline = Sequential[OutputSchema](
        data_source=MockDataSource([{"text": "test", "value": 1}]),
        config=RegressionConfig(OutputSchema)
    )
    
    def step1(data: InputSchema) -> IntermediateSchema:
        return IntermediateSchema(
            processed=data.text.upper(),
            score=float(data.value)
        )
    
    def step2(data: OutputSchema) -> OutputSchema:  # Wrong input schema
        return data
    
    pipeline.add_step(
        inference_fn=step1,
        input_schema=InputSchema,
        output_schema=IntermediateSchema
    )
    
    with pytest.raises(ValueError, match="Schema mismatch"):
        pipeline.add_step(
            inference_fn=step2,
            input_schema=OutputSchema,
            output_schema=OutputSchema
        ).build()


def test_pipeline_from_steps():
    """Test creating pipeline from list of steps."""
    def step1(data: InputSchema) -> IntermediateSchema:
        return IntermediateSchema(
            processed=data.text.upper(),
            score=float(data.value)
        )
    
    def step2(data: IntermediateSchema) -> OutputSchema:
        return OutputSchema(
            final=data.processed,
            confidence=data.score * 100
        )
    
    steps = [
        ("step1", step1, IntermediateSchema),
        ("step2", step2, OutputSchema),
    ]
    
    pipeline = Sequential.from_steps(
        data_source=MockDataSource([{"text": "test", "value": 1}]),
        steps=steps,
        config=RegressionConfig(OutputSchema)
    )
    
    assert pipeline._built
    assert len(pipeline._steps) == 2
    assert pipeline._steps[0].name == "step1"
    assert pipeline._steps[1].name == "step2"


def test_pipeline_with_mixed_steps() -> None:
    """Test pipeline with both simple and complex parameter steps."""
    # Simple step with direct Pydantic I/O
    def simple_step(data: InputSchema) -> IntermediateSchema:
        return IntermediateSchema(
            processed=data.text.upper(),
            score=float(data.value)
        )
    
    # Complex step with parameter mapping
    def complex_step(processed: str, score: float, multiplier: float) -> dict:
        return {
            "final": processed,
            "confidence": score * multiplier
        }
    
    def param_mapper(data: IntermediateSchema) -> dict:
        return {
            "processed": data.processed,
            "score": data.score,
            "multiplier": 100.0
        }
    
    pipeline = Sequential[OutputSchema](
        data_source=MockDataSource([{"text": "test", "value": 1}]),
        config=RegressionConfig(OutputSchema)
    )
    
    # Add both types of steps
    pipeline.add_step(
        inference_fn=simple_step,
        input_schema=InputSchema,
        output_schema=IntermediateSchema
    ).add_step(
        inference_fn=complex_step,
        input_schema=IntermediateSchema,
        output_schema=OutputSchema,
        arg_mapper=param_mapper
    ).build()
    
    results = pipeline.run()
    
    # Verify results
    assert len(results.step_validations) == 2
    # Could add more specific assertions about the output values


def test_pipeline_with_invalid_mapping() -> None:
    """Test pipeline fails appropriately with invalid argument mapping."""
    def step_fn(x: int, y: int) -> dict:
        return {"final": str(x + y), "confidence": 1.0}
    
    def bad_mapper(data: InputSchema) -> dict:
        return {"x": data.value}  # Missing 'y' parameter
    
    pipeline = Sequential[OutputSchema](
        data_source=MockDataSource([{"text": "test", "value": 1}]),
        config=RegressionConfig(OutputSchema)
    )
    
    pipeline.add_step(
        inference_fn=step_fn,
        input_schema=InputSchema,
        output_schema=OutputSchema,
        arg_mapper=bad_mapper
    ).build()
    
    with pytest.raises(TypeError):  # Missing required argument
        pipeline.run()