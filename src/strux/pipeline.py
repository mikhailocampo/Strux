"""Pipeline for running regression tests on model outputs."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from strux.configs import RegressionConfig
from strux.data_loading import DataSource
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


class Sequential(Pipeline[T]):
    """Pipeline that executes steps in sequence."""

    def __init__(
        self,
        data_source: DataSource,
        config: RegressionConfig[T],
        *,
        batch_size: int = 32,
        baseline_run_id: str | None = None,
    ) -> None:
        """Initialize a sequential pipeline.

        Args:
            data_source: The data source to load data from.
            steps: The steps to execute in sequence.
            config: The configuration for the regression testing.
            batch_size: The number of samples to process in each batch.
            baseline_run_id: The ID of the baseline run to compare against.

        """
        super().__init__(
            data_source=data_source,
            config=config,
            batch_size=batch_size,
            baseline_run_id=baseline_run_id,
        )

    @classmethod
    def from_steps(
        cls,
        data_source: DataSource,
        steps: list[tuple[str, Callable, type[BaseModel]]],
        config: RegressionConfig[T],
        **kwargs: dict[str, Any],
    ) -> "Sequential[T]":
        """Create a pipeline from a list of steps.

        Args:
            data_source: The data source to load data from.
            steps: List of (name, inference_fn, output_schema) tuples.
            config: The configuration for the regression testing.
            **kwargs: Additional keyword arguments for the pipeline.

        Returns:
            A new pipeline instance.

        """
        pipeline = cls(
            data_source=data_source,
            config=config,
            **kwargs,
        )
        prev_schema = data_source.schema
        for name, fn, schema in steps:
            pipeline.add_step(
                inference_fn=fn,
                input_schema=prev_schema,
                output_schema=schema,
                name=name,
            )
            prev_schema = schema

        return pipeline.build()

    def _validate_step_connections(self) -> None:
        """Validate that step input/output schemas are compatible."""
        for i in range(len(self._steps) - 1):
            current_step = self._steps[i]
            next_step = self._steps[i + 1]

            if current_step.output_schema != next_step.input_schema:
                raise ValueError(
                    f"Schema mismatch: output of '{current_step.name}' "
                    f"({current_step.output_schema.__name__}) does not match input of "
                    f"'{next_step.name}' ({next_step.input_schema.__name__})"
                )

    def run(self, baseline_path: str | None = None) -> RegressionResults[T]:
        """Run the pipeline with optional baseline comparison.

        Args:
            baseline_path: Optional path to baseline results

        """
        super().run()  # Validate built status

        # Load data and run inference
        df_data = self.data_source.load_as_df()
        step_validations = []

        for _, row in df_data.iterrows():
            current_data = row.to_dict()
            for step in self._steps:
                output_data = step.run(current_data)
                current_data = output_data.model_dump()

                # Validate outputs against config
                field_validations = []
                for field_name, field_config in self.config.field_configs.items():
                    if field_name in output_data.model_dump():
                        # For first run (no baseline), we should PASS if the field exists
                        is_first_run = baseline_path is None
                        current_value = output_data.model_dump()[field_name]

                        validation = FieldValidation(
                            field_name=field_name,
                            baseline_value=None,  # Will be populated later if baseline exists
                            current_value=current_value,
                            score=1.0 if is_first_run else 0.0,  # Pass on first run
                            threshold=field_config.threshold,
                            level=field_config.level,
                            status=ValidationStatus.PASSED if is_first_run else ValidationStatus.FAILED,
                            details={"is_first_run": is_first_run},
                        )
                        field_validations.append(validation)

                step_validations.append(
                    StepValidation(
                        step_name=step.name,
                        field_validations=field_validations,
                        metadata={"is_first_run": baseline_path is None},
                    )
                )

        results = RegressionResults(
            run_id=self._generate_run_id(),
            timestamp=datetime.now(),
            config=self.config,
            step_validations=step_validations,
        )

        # If we have a baseline, compare against it
        if baseline_path:
            try:
                baseline = RegressionResults.load_baseline(baseline_path, target_schema=self.config.target_schema)
                print(f"Loaded baseline from: {baseline_path}")
                results = results.compare_with(baseline)
            except FileNotFoundError:
                print(f"Warning: No baseline found at {baseline_path}")
                print("Running without baseline comparison")

        return results
