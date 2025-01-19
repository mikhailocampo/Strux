"""Pipeline for running regression tests on model outputs."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar, List, Tuple, Type

from pydantic import BaseModel

from strux.batch_processing import BatchProcessor
from strux.configs import RegressionConfig
from strux.data_loading import DataSource, CSVDataSource
from strux.results import FieldValidation, RegressionResults, StepValidation, ValidationStatus
from strux.step import Step

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", bound=BaseModel)


class Pipeline(ABC, Generic[T]):
    """Base class for running regression testing pipelines."""

    def __init__(
        self,
        data_source: DataSource,
        config: RegressionConfig[T],
        *,
        batch_size: int = 32,
        baseline_run_id: str | None = None,
    ) -> None:
        """Initialize regression pipeline.

        Args:
            data_source: The data source to load data from.
            config: The configuration for the regression testing.
            batch_size: The number of samples to process in each batch.
            baseline_run_id: The ID of the baseline run to compare against.

        """
        self.data_source = data_source
        self.config = config
        self.batch_size = batch_size
        self.baseline_run_id = baseline_run_id
        self._steps: list[Step] = []
        self._built = False

    def add_step(
        self,
        inference_fn: Callable[..., Any],
        input_schema: type[T],
        output_schema: type[U],
        name: str | None = None,
        description: str | None = None,
        arg_mapper: Callable[[T], dict[str, Any]] | None = None,
    ) -> "Pipeline[T]":
        """Add a step to the pipeline.

        Args:
            inference_fn: The function to use for inference.
            input_schema: The Pydantic model of the input data.
            output_schema: The Pydantic model of the output data.
            name: The name of the step.
            description: The description of the step.
            arg_mapper: An optional function to map the input data to the arguments of the inference function.

        """
        if self._built:
            raise ValueError("Cannot add steps after pipeline is built.")

        step = Step(
            name=name or f"step_{len(self._steps)}",
            inference_fn=inference_fn,
            input_schema=input_schema,
            output_schema=output_schema,
            description=description,
            arg_mapper=arg_mapper,
        )
        self._steps.append(step)
        return self

    @abstractmethod
    def _validate_step_connections(self) -> None:
        """Validate that step input/output schemas are compatible.

        Raises:
            ValueError: If step input/output schemas are incompatible.

        """

    def build(self) -> "Pipeline[T]":
        """Validate and finalize pipeline configuration.

        Returns:
            self for method chaining.

        Raises:
            ValueError: If pipeline configuration is invalid or already built.

        """
        if not self._steps:
            raise ValueError("Pipeline must have at least one step.")

        self._validate_step_connections()
        self._built = True
        return self

    def _generate_run_id(self) -> str:
        """Generate a unique run ID for the pipeline."""
        source_name = self.data_source.__class__.__name__
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{source_name}_{timestamp}"

    @abstractmethod
    def run(self) -> RegressionResults[T]:
        """Run the pipeline and return results.

        Raises:
            RuntimeError: If pipeline hadn't been built.

        """
        if not self._built:
            raise RuntimeError("Pipeline must be built before running.")


class Sequential:
    """Sequential pipeline for processing and validating data."""
    
    def __init__(
        self,
        data_source: CSVDataSource,
        config: RegressionConfig,
        steps: List[Tuple[str, Callable, Type[BaseModel]]] | None = None
    ):
        """Initialize pipeline.
        
        Args:
            data_source: Source of input data
            config: Configuration for validation
            steps: List of (name, function, output_schema) tuples
        """
        self.data_source = data_source
        self.config = config
        self.steps = steps or []

    @classmethod
    def from_steps(
        cls,
        data_source: CSVDataSource,
        steps: List[Tuple[str, Callable, Type[BaseModel]]],
        config: RegressionConfig
    ) -> 'Sequential':
        """Create pipeline from steps."""
        return cls(
            data_source=data_source,
            config=config,
            steps=steps
        )

    def _validate_batch(self, outputs: List[BaseModel]) -> List[FieldValidation]:
        """Validate a batch of outputs."""
        validations = []
        
        # Get annotation values if configured
        annotation_field = self.config.annotation_field
        annotation_values = (
            self.data_source.data[annotation_field].tolist()
            if annotation_field
            else None
        )
        
        # Validate each configured field
        for field_name, field_config in self.config.field_configs.items():
            # Get output values for this field
            output_values = [getattr(output, field_name) for output in outputs]
            
            # Compare with annotations if configured
            if field_config.compare_with_annotation and annotation_values:
                score = sum(1 for a, b in zip(output_values, annotation_values) if a == b) / len(output_values)
                baseline = annotation_values
            else:
                score = 1.0  # Default score if no comparison
                baseline = None
            
            # Create validation result
            validations.append(
                FieldValidation(
                    field_name=field_name,
                    baseline_value=baseline,
                    current_value=output_values,
                    score=score,
                    threshold=field_config.threshold,
                    level=field_config.level,
                    status=ValidationStatus.PASSED if score >= (field_config.threshold or 1.0) else ValidationStatus.FAILED,
                    details={}
                )
            )
        
        return validations

    def run(self) -> RegressionResults:
        """Run the pipeline on a batch of data."""
        # Get batch of input data
        inputs = self.data_source.get_batch()
        
        # Process each step
        for step_name, func, output_schema in self.steps:
            # Process entire batch
            outputs = [func(input_data) for input_data in inputs]
            
            # Validate outputs
            validations = self._validate_batch(outputs)
            
            # Create results
            results = RegressionResults(
                run_id=f"CSVDataSource_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(timezone.utc),
                config=self.config,
                step_validations=[
                    StepValidation(
                        step_name=step_name,
                        field_validations=validations,
                        metadata={},
                        inputs=inputs,
                        outputs=outputs
                    )
                ]
            )
        
        return results
