from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from typing import Optional, Dict, Any, TYPE_CHECKING

import plotly.graph_objects as go

from strux.visualization.plotting import create_annotation_plot, create_experiment_plot

if TYPE_CHECKING:
    from strux.experiment import Experiment
    from strux.results import RegressionResults

class HTMLReport:
    def __init__(self, template_dir: Path | None = None):
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True
        )

    def _prepare_data(self, results: 'RegressionResults') -> Dict[str, Any]:
        """Prepare data for template rendering."""
        data = {
            "fields": {},
            "inputs": [],
            "is_annotation_based": results.is_annotation_based,
            "summary": {
                "run_id": results.run_id,
                "timestamp": results.timestamp,
                "status": "PASSED" if results.passed else "FAILED",
                "total_validations": len(results.step_validations),
                "failed_validations": len([s for s in results.step_validations if not s.passed])
            }
        }
        
        # Extract field data and inputs/outputs
        for step in results.step_validations:
            # Store inputs for display
            data["inputs"].extend(
                [input_model.numbers for input_model in step.inputs]
            )
            
            for validation in step.field_validations:
                data["fields"][validation.field_name] = {
                    "name": validation.field_name,
                    "predictions": validation.current_value,
                    "annotations": validation.baseline_value,
                    "score": validation.score,
                    "threshold": validation.threshold
                }
                
        return data

    def _generate_plots(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate plots for each field."""
        plots = {}
        for field_name, field_data in data["fields"].items():
            if data.get("is_annotation_based"):
                fig = create_annotation_plot(field_data)
            else:
                fig = create_experiment_plot(field_data)
            plots[field_name] = fig.to_html(full_html=False)
        return plots

    def generate(
        self, 
        results: 'RegressionResults',
        comparison: Optional['RegressionResults'] = None,
        include_plots: bool = True
    ) -> str:
        """Generate HTML report with optional interactive plots."""
        # Prepare data for template
        data = self._prepare_data(results)
        
        # Select appropriate template
        if results.is_annotation_based:
            template = self.env.get_template("annotation_report.html")
        else:
            template = self.env.get_template("report.html")
            
        # Add plots if requested
        if include_plots:
            data["plots"] = self._generate_plots(data)
            
        return template.render(data=data) 