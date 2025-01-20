from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from typing import Optional, Dict, Any, TYPE_CHECKING
import json

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
        
        # Add utility functions to template environment
        self.env.globals.update({
            'enumerate': enumerate,
            'len': len,
            'zip': zip,
            'str': str,
        })

    def _prepare_data(self, results: 'RegressionResults') -> Dict[str, Any]:
        """Prepare data for template rendering."""
        all_validations = []
        all_inputs = []
        all_outputs = []
        
        # For comparison results, we need to get data from the baseline
        if results.baseline_results:
            print(f"Processing comparison results")  # Debug
            baseline = results.baseline_results
            
            # Process baseline results
            for step in baseline.step_validations:
                print(f"Processing baseline step: {step.step_name}")  # Debug
                all_validations.extend(step.field_validations)
                all_inputs.extend(step.inputs)
                all_outputs.extend(step.outputs)
        else:
            # Regular results processing
            for step in results.step_validations:
                print(f"Processing step: {step.step_name}")  # Debug
                all_validations.extend(step.field_validations)
                all_inputs.extend(step.inputs)
                all_outputs.extend(step.outputs)

        # Prepare field data
        fields = {}
        for validation in all_validations:
            field_data = {
                "name": validation.field_name,
                "score": validation.score,
                "threshold": validation.threshold,
                "predictions": validation.current_value,
                "annotations": validation.baseline_value,
                "details": validation.details or {},
                "plots": {}
            }
            
            # Add error analysis
            if validation.baseline_value is not None:
                if isinstance(validation.current_value[0], (int, float)):
                    # For numeric fields (like rating)
                    errors = [abs(p - a) for p, a in zip(validation.current_value, validation.baseline_value)]
                    field_data["details"].update({
                        "mae": sum(errors) / len(errors),
                        "rmse": (sum(e * e for e in errors) / len(errors)) ** 0.5,
                    })
                else:
                    # For categorical fields (like sentiment)
                    matches = sum(p == a for p, a in zip(validation.current_value, validation.baseline_value))
                    field_data["details"]["accuracy"] = matches / len(validation.current_value)
                    
                    # Add confusion matrix
                    labels = sorted(set(validation.current_value + validation.baseline_value))
                    confusion = {(true, pred): 0 for true in labels for pred in labels}
                    for p, a in zip(validation.current_value, validation.baseline_value):
                        confusion[(a, p)] += 1
                    field_data["details"]["confusion_matrix"] = confusion
            
            # Generate plot
            if results.is_annotation_based:
                plot = create_annotation_plot(field_data)
            else:
                plot = create_experiment_plot(field_data)
            
            field_data["plots"][validation.field_name] = plot.to_html(
                full_html=False,
                include_plotlyjs=False
            )
            
            fields[validation.field_name] = field_data

        # Prepare summary data
        data = {
            "summary": {
                "run_id": results.run_id,
                "timestamp": results.timestamp,
                "status": "PASSED" if results.passed else "FAILED",
                "total_validations": len(all_validations),
                "failed_validations": len([v for v in all_validations if not v.passed]),
                "is_comparison": bool(results.baseline_results)
            },
            "fields": fields,
            "inputs": [
                input_model.model_dump() if hasattr(input_model, 'model_dump') 
                else input_model 
                for input_model in all_inputs
            ],
            "outputs": [
                output_model.model_dump() if hasattr(output_model, 'model_dump')
                else output_model
                for output_model in all_outputs
            ]
        }
        
        print(f"Total validations: {len(all_validations)}")  # Debug
        print(f"Fields: {list(fields.keys())}")  # Debug
        print(f"Total inputs: {len(all_inputs)}")  # Debug
        
        return data

    def _generate_plots(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate interactive plots for the report.
        
        Returns:
            Dict mapping field names to plot HTML.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not installed, skipping plots")
            return {}

        plots = {}
        
        # Generate plots for each field
        for field_name, field_data in data.get("fields", {}).items():
            print(f"Generating plot for {field_name}")  # Debug
            
            fig = go.Figure()
            samples = field_data.get("samples", [])
            
            if not samples:
                continue
            
            # Extract values for plotting
            indices = list(range(1, len(samples) + 1))
            baseline_values = [s["baseline"] for s in samples]
            current_values = [s["current"] for s in samples]
            
            # Check if we're dealing with numeric data
            is_numeric = all(isinstance(v, (int, float)) for v in baseline_values + current_values)
            
            if is_numeric:
                # Numeric data - line plot
                fig.add_trace(go.Scatter(
                    x=indices,
                    y=baseline_values,
                    name="Baseline",
                    mode="lines+markers",
                    line=dict(color="blue")
                ))
                
                fig.add_trace(go.Scatter(
                    x=indices,
                    y=current_values,
                    name="Current",
                    mode="lines+markers",
                    line=dict(color="green")
                ))
                
                # Add status indicators
                for i, sample in enumerate(samples):
                    color = {
                        "improved": "green",
                        "unchanged": "gray",
                        "regressed": "red"
                    }.get(sample["status"], "gray")
                    
                    fig.add_trace(go.Scatter(
                        x=[i+1],
                        y=[current_values[i]],
                        mode="markers",
                        marker=dict(color=color, size=10),
                        showlegend=False
                    ))
            else:
                # Categorical data - bar plot
                categories = sorted(set(baseline_values + current_values))
                baseline_counts = [baseline_values.count(cat) for cat in categories]
                current_counts = [current_values.count(cat) for cat in categories]
                
                fig.add_trace(go.Bar(
                    x=categories,
                    y=baseline_counts,
                    name="Baseline",
                    marker_color="blue"
                ))
                
                fig.add_trace(go.Bar(
                    x=categories,
                    y=current_counts,
                    name="Current",
                    marker_color="green"
                ))

            # Update layout
            fig.update_layout(
                title=f"{field_name} Comparison",
                xaxis_title="Sample Index" if is_numeric else "Categories",
                yaxis_title="Value" if is_numeric else "Count",
                hovermode="x unified",
                showlegend=True,
                width=800,
                height=400
            )
            
            # Add metrics to the plot
            metrics_text = (
                f"Improvements: {field_data['metrics']['improvements']}<br>"
                f"Unchanged: {field_data['metrics']['unchanged']}<br>"
                f"Regressions: {field_data['metrics'].get('regressions', 0)}"
            )
            
            fig.add_annotation(
                text=metrics_text,
                xref="paper", yref="paper",
                x=1, y=1,
                showarrow=False,
                font=dict(size=12),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
            
            # Convert to HTML
            plots[field_name] = fig.to_html(
                full_html=False,
                include_plotlyjs='cdn',
                config={'displayModeBar': False}
            )
        
        return plots

    def _prepare_comparison_data(self, results: 'RegressionResults') -> Dict[str, Any]:
        """Prepare data specifically for comparison reports."""
        baseline = results.baseline_results
        if not baseline:
            print("No baseline results found!")  # Debug
            return {}

        print(f"\nPreparing comparison data:")  # Debug
        
        # Get run IDs from metadata
        baseline_run_id = baseline.metadata.get("run_id", "baseline")
        current_run_id = results.metadata.get("run_id", "current")
        print(f"Baseline run: {baseline_run_id}")
        print(f"Current run: {current_run_id}")

        print(f"\nAnnotation config:")
        print(f"Annotation field: {results.config.annotation_field}")
        print(f"Field configs: {results.config.field_configs}")

        comparison_data = {
            "summary": {
                "run_id": current_run_id,
                "baseline_id": baseline_run_id,
                "timestamp": results.timestamp,
                "status": "PASSED" if results.passed else "FAILED",
            },
            "fields": {},
            "changes": {
                "improvements": 0,
                "regressions": 0,
                "unchanged": 0
            }
        }

        # Get diff validations from metadata
        diff_validations = results.metadata.get("diff_validations", [])

        # Get annotations from the data source
        annotations = []
        if (results.config.annotation_field and 
            results.config.data_source and 
            hasattr(results.config.data_source, "data")):
            
            # Get the parsed annotations directly from the data source
            raw_data = results.config.data_source.data
            if results.config.annotation_field in raw_data.columns:
                for annotation in raw_data[results.config.annotation_field]:
                    if isinstance(annotation, results.config.target_schema):
                        # Already parsed annotation
                        annotations.append(annotation)
                    elif isinstance(annotation, dict):
                        try:
                            annotations.append(results.config.target_schema(**annotation))
                        except ValueError as e:
                            print(f"Error parsing annotation dict: {e}")
                            annotations.append(None)
                    elif isinstance(annotation, str):
                        try:
                            annotation_data = json.loads(annotation)
                            annotations.append(results.config.target_schema(**annotation_data))
                        except (json.JSONDecodeError, ValueError) as e:
                            print(f"Error parsing annotation: {e}")
                            annotations.append(None)
                    else:
                        print(f"Unknown annotation type: {type(annotation)}")
                        annotations.append(None)
        
        print(f"Found {len(annotations)} annotations")
        # Print first annotation for debugging
        if annotations:
            print(f"First annotation: {annotations[0]}")

        # Process each step validation
        for step_validation in diff_validations:
            for field_validation in step_validation.field_validations:
                field_name = field_validation.field_name
                print(f"\nAnalyzing field: {field_name}")  # Debug

                field_data = {
                    "name": field_name,
                    "baseline_score": field_validation.score,
                    "current_score": field_validation.score,
                    "samples": [],
                    "metrics": {
                        "improvements": 0,
                        "regressions": 0,
                        "unchanged": 0
                    }
                }

                # Compare each sample
                for idx, (baseline_val, current_val) in enumerate(
                    zip(field_validation.baseline_value, field_validation.current_value)
                ):
                    print(f"\nSample {idx + 1}:")  # Debug
                    print(f"Baseline: {baseline_val}")
                    print(f"Current: {current_val}")
                    
                    # Get expected value from annotation if available
                    expected = None
                    if idx < len(annotations) and annotations[idx] is not None:
                        expected = getattr(annotations[idx], field_name, None)
                        print(f"Expected: {expected}")
                    
                    # Determine if this is an improvement/regression
                    if baseline_val == current_val:
                        status = "unchanged"
                        field_data["metrics"]["unchanged"] += 1
                        comparison_data["changes"]["unchanged"] += 1
                    elif expected is not None:
                        # Compare against annotation if available
                        baseline_correct = baseline_val == expected
                        current_correct = current_val == expected
                        
                        if current_correct and not baseline_correct:
                            status = "improved"
                            field_data["metrics"]["improvements"] += 1
                            comparison_data["changes"]["improvements"] += 1
                        elif baseline_correct and not current_correct:
                            status = "regressed"
                            field_data["metrics"]["regressions"] += 1
                            comparison_data["changes"]["regressions"] += 1
                        else:
                            status = "unchanged"
                            field_data["metrics"]["unchanged"] += 1
                            comparison_data["changes"]["unchanged"] += 1
                    else:
                        # Without annotation, treat any change as improvement
                        status = "improved"
                        field_data["metrics"]["improvements"] += 1
                        comparison_data["changes"]["improvements"] += 1

                    # Store sample data
                    field_data["samples"].append({
                        "index": idx + 1,
                        "baseline": baseline_val,
                        "current": current_val,
                        "expected": expected,
                        "status": status,
                        "input": step_validation.inputs[idx] if step_validation.inputs else None
                    })

                print(f"\nField metrics: {field_data['metrics']}")  # Debug
                comparison_data["fields"][field_name] = field_data

        print(f"\nOverall changes: {comparison_data['changes']}")  # Debug
        return comparison_data

    def generate(
        self, 
        results: 'RegressionResults',
        comparison: Optional['RegressionResults'] = None,
        include_plots: bool = True
    ) -> str:
        """Generate HTML report with optional interactive plots."""
        if results.baseline_results:
            data = self._prepare_comparison_data(results)
            template = self.env.get_template("comparison_report.html")
        else:
            data = self._prepare_data(results)
            template = self.env.get_template(
                "annotation_report.html" if results.is_annotation_based 
                else "experiment_report.html"
            )
        
        if include_plots:
            data["plots"] = self._generate_plots(data)
        
        return template.render(data=data) 