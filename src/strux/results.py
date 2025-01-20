"""Storage and analysis of regression testing results."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from strux.configs import FieldConfig, RegressionConfig, ValidationLevel
from strux.strategies import ComparisonStrategy, ValidationStrategy
from strux.types import T, MetadataDict

if TYPE_CHECKING:
    from strux.visualization.report_generator import HTMLReport

T = TypeVar("T", bound=BaseModel)


class ValidationStatus(Enum):
    """Status of a validation result."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class FieldValidation:
    """Results of validating a single field."""

    field_name: str
    current_value: List[Any]
    baseline_value: Optional[List[Any]]
    score: float
    threshold: float
    details: Dict[str, Any]

    @property
    def passed(self) -> bool:
        """Check if validation passed threshold."""
        return self.score >= self.threshold

    @property
    def status(self) -> ValidationStatus:
        """Get validation status."""
        return ValidationStatus.PASSED if self.passed else ValidationStatus.FAILED


@dataclass
class StepValidation:
    """Results of validating a single step."""

    step_name: str
    field_validations: List[FieldValidation]
    metadata: Dict[str, Any]
    inputs: List[Any]
    outputs: List[Any]

    @property
    def passed(self) -> bool:
        """Check if all field validations passed."""
        return all(v.passed for v in self.field_validations)

    @property
    def status(self) -> ValidationStatus:
        """Calculate overall status based on field validations."""
        return ValidationStatus.PASSED if self.passed else ValidationStatus.FAILED

    def get_failed_validations(self) -> list[FieldValidation]:
        """Get all failed field validations."""
        return [v for v in self.field_validations if v.status != ValidationStatus.PASSED]

    def format_summary(self) -> str:
        """Format validation results as text summary."""
        lines = [f"Step: {self.step_name}"]
        lines.append(f"Status: {self.status.value}")
        
        # Format each field validation
        for validation in self.field_validations:
            lines.append(f"\nField: {validation.field_name}")
            lines.append(f"Score: {validation.score:.2f} (threshold: {validation.threshold})")
            
            # Show sample of predictions vs annotations
            for i, (pred, baseline) in enumerate(zip(
                validation.current_value, 
                validation.baseline_value or []
            )):
                lines.append(f"\n  Row {i+1}:")
                lines.append(f"    Predicted: {pred}")
                if baseline:
                    lines.append(f"    Expected:  {baseline}")
                
        return "\n".join(lines)


class RegressionResults(BaseModel, Generic[T]):
    """Container for regression testing results."""

    run_id: str
    timestamp: datetime
    config: RegressionConfig[T]
    step_validations: list[StepValidation]
    metadata: dict[str, Any] = Field(default_factory=dict)
    _baseline_results: Optional["RegressionResults[T]"] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        run_id: str,
        timestamp: datetime,
        config: RegressionConfig[T],
        step_validations: list[StepValidation],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize regression results."""
        super().__init__(
            run_id=run_id,
            timestamp=timestamp,
            config=config,
            step_validations=step_validations,
            metadata=metadata or {},
        )

    @property
    def passed(self) -> bool:
        """Check if all steps passed validation."""
        return all(v.passed for v in self.step_validations)

    def get_step_validations(self, step_name: str) -> StepValidation | None:
        """Get validation results for a specific step."""
        return next((v for v in self.step_validations if v.step_name == step_name), None)

    def get_failed_steps(self) -> list[StepValidation]:
        """Get all steps that failed validation."""
        return [v for v in self.step_validations if not v.passed]

    def format_summary(self, *, include_passing: bool = True) -> str:
        """Generate human-readable summary of results.

        Args:
            include_passing: Whether to include passing validations

        """
        lines = [
            f"Run ID: {self.run_id}",
            f"Timestamp: {self.timestamp}",
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            "",
            "Results:",
        ]

        for step in self.step_validations:
            if include_passing or step.status != ValidationStatus.PASSED:
                lines.append(step.format_summary())
                lines.append("")  # Empty line between steps

        return "\n".join(lines)

    def compare_with(self, baseline: "RegressionResults[T]") -> "RegressionResults[T]":
        """Compare current results with a baseline run."""
        print(f"\nComparing results:")  # Debug
        print(f"Baseline steps: {[s.step_name for s in baseline.step_validations]}")
        print(f"Current steps: {[s.step_name for s in self.step_validations]}")
        
        diff_validations = []
        current_validations = []

        # Create step mapping first
        step_mapping = {}
        for current_step, baseline_step in zip(self.step_validations, baseline.step_validations):
            step_mapping[current_step.step_name] = baseline_step.step_name
            if current_step.step_name != baseline_step.step_name:
                print(f"No exact step match, using positional match")
                print(f"Matching {baseline_step.step_name} with {current_step.step_name}")

        # Match steps by position when names don't match
        for current, baseline in zip(self.step_validations, baseline.step_validations):
            field_validations = []
            for current_field in current.field_validations:
                # Find matching baseline field
                baseline_field = next(
                    (f for f in baseline.field_validations if f.field_name == current_field.field_name),
                    None
                )
                
                if not baseline_field:
                    continue
                    
                print(f"Comparing field {current_field.field_name}")
                print(f"Baseline values: {baseline_field.current_value}")
                print(f"Current values: {current_field.current_value}")
                
                # Get strategy from config
                field_config = self.config.field_configs.get(current_field.field_name)
                if not field_config or not field_config.strategy:
                    print(f"No strategy configured for field {current_field.field_name}, skipping")
                    continue
                    
                strategy = field_config.strategy
                score = 0.0
                
                # Handle different strategy types
                if isinstance(strategy, ValidationStrategy):
                    score = strategy.validate(
                        predictions=current_field.current_value,
                        annotations=baseline_field.current_value
                    )
                elif isinstance(strategy, ComparisonStrategy):
                    scores = []
                    for baseline_val, current_val in zip(baseline_field.current_value, current_field.current_value):
                        scores.append(strategy.compare(baseline_val, current_val))
                    score = sum(scores) / len(scores) if scores else 0.0
                else:
                    print(f"Unknown strategy type for field {current_field.field_name}: {type(strategy)}")
                    continue
                
                validation = FieldValidation(
                    field_name=current_field.field_name,
                    current_value=current_field.current_value,
                    baseline_value=baseline_field.current_value,
                    score=score,
                    threshold=current_field.threshold,
                    details={
                        "baseline_step": baseline.step_name,
                        "current_step": current.step_name,
                        "strategy": type(strategy).__name__
                    }
                )
                field_validations.append(validation)
            
            if field_validations:
                step_validation = StepValidation(
                    step_name=current.step_name,
                    field_validations=field_validations,
                    metadata={
                        "baseline_step": baseline.step_name,
                        "current_step": current.step_name,
                    },
                    inputs=current.inputs,
                    outputs=current.outputs
                )
                diff_validations.append(step_validation)
                current_validations.append(current)

        # Get run IDs from metadata if available, otherwise generate them
        baseline_run_id = baseline.metadata.get("run_id", "baseline")
        current_run_id = self.metadata.get("run_id", "current")
        
        # Create comparison results
        results = RegressionResults(
            run_id=f"diff_{current_run_id}_vs_{baseline_run_id}",
            timestamp=datetime.now(timezone.utc),
            config=self.config,
            step_validations=current_validations,
            metadata={
                "baseline_run_id": baseline_run_id,
                "current_run_id": current_run_id,
                "diff_validations": diff_validations,
                "step_mapping": step_mapping
            },
        )
        results._baseline_results = baseline
        return results

    @property
    def baseline_results(self) -> Optional["RegressionResults[T]"]:
        """Get the baseline results if this is a comparison."""
        return self._baseline_results

    def export(self, path: str) -> None:
        """Export results to a file.

        Args:
            path: Where to save the results

        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

        def serialize_value(obj: Any) -> Any:
            """Helper to serialize Pydantic models and other objects."""
            if isinstance(obj, BaseModel):
                return obj.model_dump()
            if isinstance(obj, (datetime, timezone)):
                return obj.isoformat()
            return obj

        # Convert to serializable format
        data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "config": {
                "target_schema": self.config.target_schema.__name__,
                "field_configs": {
                    name: {
                        "threshold": config.threshold,
                        "level": config.level.value,
                        "strategy": config.strategy.__class__.__name__ if config.strategy else None,
                    }
                    for name, config in self.config.field_configs.items()
                },
            },
            "steps": [
                {
                    "name": step.step_name,
                    "validations": [
                        {
                            "field": v.field_name,
                            "score": v.score,
                            "threshold": v.threshold,
                            "passed": v.status == ValidationStatus.PASSED,
                            "baseline": [serialize_value(x) for x in (v.baseline_value or [])],
                            "current": [serialize_value(x) for x in v.current_value],
                            "details": v.details,
                        }
                        for v in step.field_validations
                    ],
                    "metadata": step.metadata,
                    "inputs": [serialize_value(x) for x in step.inputs],
                    "outputs": [serialize_value(x) for x in step.outputs],
                }
                for step in self.step_validations
            ],
            "metadata": self.metadata,
        }

        output_path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load_baseline(cls, path: str, target_schema: type[T]) -> "RegressionResults[T]":
        """Load baseline results from a file.

        Args:
            path: Path to the baseline results file
            target_schema: The schema class used for validation

        """
        baseline_path = Path(path)
        if not baseline_path.exists():
            raise FileNotFoundError(f"No baseline found at {path}")

        data = json.loads(baseline_path.read_text())

        # Reconstruct field validations
        step_validations = []
        for step in data["steps"]:
            field_validations = [
                FieldValidation(
                    field_name=v["field"],
                    baseline_value=v["baseline"],
                    current_value=v["current"],
                    score=v["score"],
                    threshold=v["threshold"],
                    details=v.get("details", {}),
                )
                for v in step["validations"]
            ]

            step_validations.append(
                StepValidation(
                    step_name=step["name"],
                    field_validations=field_validations,
                    metadata=step.get("metadata", {}),
                    inputs=step.get("inputs", []),
                    outputs=step.get("outputs", []),
                )
            )

        # Create config with actual schema class
        config = RegressionConfig(
            target_schema=target_schema,
            field_configs={name: FieldConfig(**cfg) for name, cfg in data["config"]["field_configs"].items()},
        )

        return cls(
            run_id=data["run_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            config=config,
            step_validations=step_validations,
            metadata=data.get("metadata", {}),
        )

    def save_as_baseline(self, path: str) -> None:
        """Save these results as a baseline for future comparisons."""
        self.export(path)
        print(f"\nBaseline saved to: {path}")
        print("Use this baseline in future runs with:")
        print(f"pipeline.run(baseline_path='{path}')")

    def to_html(self, output_path: str | None = None) -> str:
        """Generate an HTML report of the results."""
        # Import HTMLReport here to avoid circular import
        from strux.visualization.report_generator import HTMLReport
        
        report = HTMLReport()
        html = report.generate(self)
        
        if output_path:
            Path(output_path).write_text(html)
            print(f"\nReport saved to: {output_path}")

        return html

    @property
    def is_annotation_based(self) -> bool:
        """Check if this result uses annotations for comparison."""
        return bool(self.config.annotation_field)

    def get_annotation_field(self) -> str | None:
        """Get the name of the annotation field if configured."""
        return self.config.annotation_field

    def get_field_values(self, field_name: str) -> tuple[list[Any], list[Any] | None]:
        """Get current and annotation values for a field.
        
        Returns:
            Tuple of (current_values, annotation_values)
        """
        current_values = []
        annotation_values = []
        
        for step in self.step_validations:
            for validation in step.field_validations:
                if validation.field_name == field_name:
                    current_values.extend(validation.current_value)
                    if validation.baseline_value is not None:
                        annotation_values.extend(validation.baseline_value)
                        
        return current_values, annotation_values if annotation_values else None
