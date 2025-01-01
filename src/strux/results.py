"""Storage and analysis of regression testing results."""

from enum import Enum
import json
from typing import Any, Dict, Generic, List, TypeVar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from strux.configs import RegressionConfig, ValidationLevel

T = TypeVar("T", bound=BaseModel)


class ValidationStatus(Enum):
    """Status of a validation result."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class FieldValidation:
    """Results of validating a single field."""

    field_name: str
    baseline_value: Any
    current_value: Any
    score: float
    threshold: float | None
    level: ValidationLevel
    status: ValidationStatus
    details: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Check if validation passed."""
        return self.status in (ValidationStatus.PASSED, ValidationStatus.SKIPPED)


@dataclass
class StepValidation:
    """Results of validating a single step."""

    step_name: str
    field_validations: list[FieldValidation]
    metadata: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Check if all field validations passed."""
        return all(v.status == ValidationStatus.PASSED for v in self.field_validations)

    def get_failed_validations(self) -> List[FieldValidation]:
        """Get all failed field validations."""
        return [v for v in self.field_validations if v.status != ValidationStatus.PASSED]

    def format_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [f"Step: {self.step_name}"]
        for validation in self.field_validations:
            status = "✓" if validation.status == ValidationStatus.PASSED else "✗"
            lines.append(
                f"  {status} {validation.field_name}: "
                f"score={validation.score:.2f} "
                f"(threshold={validation.threshold or 1.0})"
            )
            if validation.status != ValidationStatus.PASSED:
                lines.append(f"    Baseline: {validation.baseline_value}")
                lines.append(f"    Current:  {validation.current_value}")
        return "\n".join(lines)


class RegressionResults(BaseModel, Generic[T]):
    """Container for regression testing results."""
    
    run_id: str
    timestamp: datetime
    config: RegressionConfig[T]
    step_validations: List[StepValidation]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        run_id: str,
        timestamp: datetime,
        config: RegressionConfig[T],
        step_validations: List[StepValidation],
        metadata: Dict[str, Any] = None,
        status: ValidationStatus = None,
    ) -> None:
        """Initialize regression results.

        Args:
            run_id: Unique identifier for the run
            timestamp: When the run occurred
            config: Configuration used for validation
            step_validations: Results from each step
            metadata: Additional run information
            status: Status of the run
        """
        super().__init__(
            run_id=run_id,
            timestamp=timestamp,
            config=config,
            step_validations=step_validations,
            metadata=metadata or {},
            status=status,
        )

    @property
    def passed(self) -> bool:
        """Check if all steps passed validation."""
        return all(v.passed for v in self.step_validations)

    def get_step_validations(self, step_name: str) -> StepValidation:
        """Get validation results for a specific step."""
        return next(
            (v for v in self.step_validations if v.step_name == step_name), None
        )

    def get_failed_steps(self) -> List[StepValidation]:
        """Get all steps that failed validation."""
        return [v for v in self.step_validations if not v.passed]

    def format_summary(self, include_passing: bool = True) -> str:
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
        """Compare current results with a baseline run.

        Args:
            baseline: Previous run to compare against

        Returns:
            New RegressionResults containing only the differences
        """
        diff_validations = []

        for current_step in self.step_validations:
            baseline_step = baseline.get_step_validation(current_step.step_name)
            if not baseline_step:
                continue

            # Compare field validations
            diff_fields = []
            for current_field in current_step.field_validations:
                baseline_field = next(
                    (
                        f
                        for f in baseline_step.field_validations
                        if f.field_name == current_field.field_name
                    ),
                    None,
                )
                if baseline_field and current_field.score != baseline_field.score:
                    diff_fields.append(current_field)

            if diff_fields:
                diff_validations.append(
                    StepValidation(
                        step_name=current_step.step_name,
                        field_validations=diff_fields,
                        metadata={"compared_with": baseline.run_id},
                    )
                )

        return RegressionResults(
            run_id=f"diff_{self.run_id}_vs_{baseline.run_id}",
            timestamp=datetime.now(),
            config=self.config,
            step_validations=diff_validations,
            metadata={
                "baseline_run_id": baseline.run_id,
                "current_run_id": self.run_id,
            },
        )

    def export(self, path: str) -> None:
        """Export results to a file.

        Args:
            path: Where to save the results
        """
        output_path = Path(path)

        # Convert to serializable format
        data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "config": self.config.model_dump(),
            "steps": [
                {
                    "name": step.step_name,
                    "status": step.status,
                    "validations": [
                        {
                            "field": v.field_name,
                            "score": v.score,
                            "threshold": v.threshold,
                            "passed": v.status == ValidationStatus.PASSED,
                            "baseline": v.baseline_value,
                            "current": v.current_value,
                            "details": v.details,
                        }
                        for v in step.field_validations
                    ],
                    "metadata": step.metadata,
                }
                for step in self.step_validations
            ],
            "metadata": self.metadata,
        }

        output_path.write_text(json.dumps(data, indent=2))
