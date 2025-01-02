"""Step abstraction for pipeline processing representing the smallest single unit of work."""

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", bound=BaseModel)


class Step(Generic[T, U]):
    """Represents a single inference step in a pipeline.

    Generic Parameters:
        T: Input type (must be a pydantic model)
        U: Output type (must be a pydantic model)
    """

    def __init__(
        self,
        name: str,
        inference_fn: Callable[..., U],
        input_schema: type[T],
        output_schema: type[U],
        description: str | None = None,
        arg_mapper: Callable[[T], dict] | None = None,
    ) -> None:
        """Initialize a pipeline step.

        Args:
            name: The name of the step.
            inference_fn: The function to use for inference.
            input_schema: The schema of the input data.
            output_schema: The schema of the output data.
            description: A description of the step.
            arg_mapper: An optional function to map the input data to the arguments of the inference function.

        """
        self.name = name
        self.inference_fn = inference_fn
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.description = description or f"Step: {name}"
        self.arg_mapper = arg_mapper

    def run(self, input_data: Any) -> U:
        """Execute the step's inference function with validation.

        Args:
            input_data: Input data to process

        Returns:
            Validated output from the inference function

        Raises:
            ValidationError: If input or output validation fails.

        """
        # Validate input
        validated_input = self.input_schema.model_validate(input_data)

        if self.arg_mapper:
            # Use arg mapper for complex parameter mapping
            kwargs = self.arg_mapper(validated_input)
            raw_output = self.inference_fn(**kwargs)
        else:
            # Default: pass validated input directly to inference function
            raw_output = self.inference_fn(validated_input)

        # Validate output
        validated_output = self.output_schema.model_validate(raw_output)

        return validated_output
