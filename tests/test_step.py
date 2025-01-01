"""Tests for pipeline steps."""

import pytest
from pydantic import BaseModel

from strux.step import Step


class InputModel(BaseModel):
    text: str
    value: int


class OutputModel(BaseModel):
    processed: str
    score: float


def sample_inference(input_data: InputModel) -> OutputModel:
    """Sample inference function."""
    return OutputModel(
        processed=input_data.text.upper(),
        score=float(input_data.value) * 1.5,
    )


def test_step_initialization() -> None:
    """Test basic step initialization."""
    step = Step(
        name="sample_step",
        inference_fn=sample_inference,
        input_schema=InputModel,
        output_schema=OutputModel,
        description="Sample step for testing",
    )
    assert step.name == "sample_step"
    assert step.description == "Sample step for testing"
    assert step.input_schema == InputModel
    assert step.output_schema == OutputModel


def test_step_inference() -> None:
    """Test running inference through a step."""
    step = Step(
        name="test_step",
        inference_fn=sample_inference,
        input_schema=InputModel,
        output_schema=OutputModel
    )
    
    input_data = InputModel(text="hello", value=10)
    result = step.run(input_data)
    
    assert isinstance(result, OutputModel)
    assert result.processed == "HELLO"
    assert result.score == 15.0


def test_invalid_input() -> None:
    """Test step with invalid input data."""
    step = Step(
        name="test_step",
        inference_fn=sample_inference,
        input_schema=InputModel,
        output_schema=OutputModel
    )
    
    with pytest.raises(ValueError):
        step.run({"text": "hello"})  # Missing value field


def test_invalid_output() -> None:
    """Test step with invalid output from inference."""
    def bad_inference(_: InputModel) -> dict:
        return {"wrong": "output"}
    
    step = Step(
        name="test_step",
        inference_fn=bad_inference,
        input_schema=InputModel,
        output_schema=OutputModel
    )
    
    with pytest.raises(ValueError):
        step.run(InputModel(text="hello", value=10))


def test_step_with_arg_mapper() -> None:
    """Test step with custom argument mapping."""
    def complex_inference(text: str, value: int, multiplier: float = 1.0) -> dict:
        return {
            "processed": text.upper(),
            "score": value * multiplier
        }
    
    def arg_mapper(data: InputModel) -> dict:
        return {
            "text": data.text,
            "value": data.value,
            "multiplier": 1.5
        }
    
    step = Step(
        name="complex_step",
        inference_fn=complex_inference,
        input_schema=InputModel,
        output_schema=OutputModel,
        arg_mapper=arg_mapper
    )
    
    result = step.run(InputModel(text="hello", value=10))
    assert isinstance(result, OutputModel)
    assert result.processed == "HELLO"
    assert result.score == 15.0


def test_step_with_invalid_arg_mapping() -> None:
    """Test step with invalid argument mapping."""
    def complex_inference(text: str, value: int) -> dict:
        return {
            "processed": text.upper(),
            "score": float(value)
        }
    
    def bad_mapper(data: InputModel) -> dict:
        return {"wrong": "args"}  # Missing required arguments
    
    step = Step(
        name="bad_step",
        inference_fn=complex_inference,
        input_schema=InputModel,
        output_schema=OutputModel,
        arg_mapper=bad_mapper
    )
    
    with pytest.raises(TypeError):  # Missing required arguments
        step.run(InputModel(text="hello", value=10))