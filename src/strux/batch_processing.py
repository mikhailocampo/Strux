# src/strux/batch_processing.py
from dataclasses import dataclass
from typing import Any, Generic
import pandas as pd

from strux.types import T, MetadataDict
from strux.step import Step
from strux.configs import RegressionConfig, ValidationLevel
from strux.results import FieldValidation, StepValidation, ValidationStatus

@dataclass
class BatchResults:
    """Results for a batch of data."""
    batch_id: str
    step_validations: list[StepValidation]
    raw_outputs: dict[str, Any]
    metadata: MetadataDict

class BatchProcessor(Generic[T]):
    """Processes data in batches and validates outputs."""
    
    def __init__(self, batch_size: int = 32):
        """Initialize batch processor.
        
        Args:
            batch_size: Number of samples to process in each batch
        """
        self.batch_size = batch_size
        self.validate_outputs = self._validate_outputs
        
    def _validate_outputs(
        self,
        outputs: list[Any],
        config: RegressionConfig[T],
        is_first_run: bool,
    ) -> list[StepValidation]:
        """Validate outputs against configuration."""
        validations = []
        
        # Get annotations from metadata if available
        annotations = getattr(config.data_source, "metadata", {}).get("annotations", [])
        
        for field_name, field_config in config.field_configs.items():
            field_values = [getattr(output, field_name) for output in outputs]
            
            if is_first_run:
                validation = FieldValidation(
                    field_name=field_name,
                    baseline_value=None,
                    current_value=field_values,
                    score=1.0,
                    threshold=field_config.threshold,
                    level=field_config.level,
                    status=ValidationStatus.PASSED,
                    details={"is_first_run": True}
                )
            else:
                # Get expected values from annotations if available
                if field_config.compare_with_annotation and annotations:
                    expected_values = [
                        getattr(annotation, field_name) if annotation else None 
                        for annotation in annotations
                    ]
                else:
                    expected_values = field_values  # Self-comparison
                
                # Calculate similarity score
                if field_config.strategy:
                    score = field_config.strategy.compare(expected_values, field_values)
                else:
                    score = float(all(exp == val for exp, val in zip(expected_values, field_values)))
                
                validation = FieldValidation(
                    field_name=field_name,
                    baseline_value=expected_values,
                    current_value=field_values,
                    score=score,
                    threshold=field_config.threshold,
                    level=field_config.level,
                    status=ValidationStatus.PASSED if score >= field_config.threshold else ValidationStatus.FAILED,
                    details={}
                )
            
            validations.append(validation)
        
        return [StepValidation(
            step_name="batch_validation",
            field_validations=validations,
            metadata={"is_first_run": is_first_run}
        )]
    
    def process_batch(
        self,
        batch_df: pd.DataFrame,
        steps: list[Step],
        config: RegressionConfig[T],
        is_first_run: bool = False
    ) -> BatchResults:
        """Process a batch of data through all steps."""
        step_validations = []
        raw_outputs = {}
        
        # Try vectorized processing if supported
        for step in steps:
            if hasattr(step.inference_fn, "batch_process"):
                outputs = step.inference_fn(batch_df)
            else:
                # Fall back to row-by-row
                outputs = batch_df.apply(
                    lambda row: step.run(row.to_dict()), 
                    axis=1
                ).to_list()
            
            # Store raw outputs for later diff computation
            raw_outputs[step.name] = outputs
            
            # Validate outputs using the instance method
            validations = self.validate_outputs(
                outputs, config, is_first_run
            )
            step_validations.extend(validations)
        
        return BatchResults(
            batch_id=f"batch_{len(raw_outputs)}",
            step_validations=step_validations,
            raw_outputs=raw_outputs,
            metadata={"is_first_run": is_first_run}
        )