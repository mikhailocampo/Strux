"""Storage and analysis of regression testing results."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from strux.configs import FieldConfig, RegressionConfig, ValidationLevel

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

    def get_failed_validations(self) -> list[FieldValidation]:
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
    step_validations: list[StepValidation]
    metadata: dict[str, Any] = Field(default_factory=dict)

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
        """Compare current results with a baseline run.

        Args:
            baseline: Previous run to compare against

        Returns:
            New RegressionResults containing the differences

        """
        diff_validations = []

        for current_step in self.step_validations:
            baseline_step = baseline.get_step_validations(current_step.step_name)
            if not baseline_step:
                continue

            # Compare field validations
            field_validations = []
            for current_field in current_step.field_validations:
                baseline_field = next(
                    (f for f in baseline_step.field_validations if f.field_name == current_field.field_name), None
                )

                if baseline_field:
                    # Use the strategy from config to compare values
                    strategy = self.config.field_configs[current_field.field_name].strategy
                    score = strategy.compare(baseline_field.current_value, current_field.current_value)

                    validation = FieldValidation(
                        field_name=current_field.field_name,
                        baseline_value=baseline_field.current_value,  # Use the baseline's current value
                        current_value=current_field.current_value,
                        score=score,
                        threshold=current_field.threshold,
                        level=current_field.level,
                        status=ValidationStatus.PASSED
                        if score >= (current_field.threshold or 1.0)
                        else ValidationStatus.FAILED,
                        details={"compared_with": baseline.run_id},
                    )
                    field_validations.append(validation)

            if field_validations:
                diff_validations.append(
                    StepValidation(
                        step_name=current_step.step_name,
                        field_validations=field_validations,
                        metadata={"compared_with": baseline.run_id},
                    )
                )

        return RegressionResults(
            run_id=f"diff_{self.run_id}_vs_{baseline.run_id}",
            timestamp=datetime.now(timezone.utc),
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
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

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
                    level=ValidationLevel(v.get("level", "strict")),
                    status=ValidationStatus(v.get("status", "failed")),
                    details=v.get("details", {}),
                )
                for v in step["validations"]
            ]

            step_validations.append(
                StepValidation(
                    step_name=step["name"], field_validations=field_validations, metadata=step.get("metadata", {})
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
