import plotly.graph_objects as go
from typing import Dict, Any
import plotly.express as px

def create_annotation_plot(field_data: Dict[str, Any]) -> go.Figure:
    """Create comparison plot between predictions and annotations."""
    fig = go.Figure()
    
    predictions = field_data.get("predictions", [])
    annotations = field_data.get("annotations", [])
    
    if not predictions:
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    # For categorical data (like sentiment)
    if isinstance(predictions[0], str):
        # Create confusion matrix data
        if annotations:
            labels = sorted(set(predictions + annotations))
            confusion = {
                (true, pred): 0 
                for true in labels 
                for pred in labels
            }
            for p, a in zip(predictions, annotations):
                confusion[(a, p)] += 1
                
            # Plot confusion matrix
            fig = px.imshow(
                [[confusion[(t, p)] for p in labels] for t in labels],
                x=labels,
                y=labels,
                labels=dict(x="Predicted", y="Actual"),
                title=f"{field_data['name']} Confusion Matrix"
            )
            
            # Add accuracy score
            accuracy = sum(p == a for p, a in zip(predictions, annotations)) / len(predictions)
            fig.add_annotation(
                text=f"Accuracy: {accuracy:.2%}",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False
            )
    else:
        # For numeric data (like ratings)
        fig.add_trace(go.Scatter(
            x=list(range(len(predictions))),
            y=predictions,
            name="Predictions",
            mode="lines+markers"
        ))
        
        if annotations:
            fig.add_trace(go.Scatter(
                x=list(range(len(annotations))),
                y=annotations,
                name="Ground Truth",
                mode="lines+markers"
            ))
            
            # Add error distribution
            errors = [abs(p - a) for p, a in zip(predictions, annotations)]
            fig.add_trace(go.Box(
                y=errors,
                name="Error Distribution",
                yaxis="y2"
            ))
            
            # Add MAE and RMSE
            import numpy as np
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean([e**2 for e in errors]))
            
            fig.add_annotation(
                text=f"MAE: {mae:.2f}<br>RMSE: {rmse:.2f}",
                xref="paper", yref="paper",
                x=1.15, y=0.5,
                showarrow=False
            )
    
    fig.update_layout(
        title=f"{field_data['name']} Analysis",
        showlegend=True,
        yaxis2=dict(
            title="Error",
            overlaying="y",
            side="right"
        )
    )
    
    return fig

def create_experiment_plot(field_data: Dict[str, Any]) -> go.Figure:
    """Create comparison plot between experiment runs."""
    fig = go.Figure()
    
    current_values = field_data.get("current", [])
    baseline_values = field_data.get("baseline", [])
    
    if not current_values:
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    # For categorical data
    if isinstance(current_values[0], str):
        # Create distribution comparison
        labels = sorted(set(current_values + (baseline_values or [])))
        current_dist = {label: current_values.count(label) for label in labels}
        
        fig.add_trace(go.Bar(
            x=labels,
            y=[current_dist[label] for label in labels],
            name="Current Run"
        ))
        
        if baseline_values:
            baseline_dist = {label: baseline_values.count(label) for label in labels}
            fig.add_trace(go.Bar(
                x=labels,
                y=[baseline_dist[label] for label in labels],
                name="Baseline"
            ))
            
            # Add agreement score
            agreement = sum(c == b for c, b in zip(current_values, baseline_values)) / len(current_values)
            fig.add_annotation(
                text=f"Agreement with baseline: {agreement:.2%}",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False
            )
    else:
        # For numeric data
        fig.add_trace(go.Scatter(
            x=list(range(len(current_values))),
            y=current_values,
            name="Current Run",
            mode="lines+markers"
        ))
        
        if baseline_values:
            fig.add_trace(go.Scatter(
                x=list(range(len(baseline_values))),
                y=baseline_values,
                name="Baseline",
                mode="lines+markers"
            ))
            
            # Add difference distribution
            diffs = [c - b for c, b in zip(current_values, baseline_values)]
            fig.add_trace(go.Box(
                y=diffs,
                name="Differences",
                yaxis="y2"
            ))
            
            # Add statistics
            import numpy as np
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            
            fig.add_annotation(
                text=f"Mean Diff: {mean_diff:.2f}<br>Std Dev: {std_diff:.2f}",
                xref="paper", yref="paper",
                x=1.15, y=0.5,
                showarrow=False
            )
    
    fig.update_layout(
        title=f"{field_data['name']} Experiment Comparison",
        showlegend=True,
        yaxis2=dict(
            title="Differences",
            overlaying="y",
            side="right"
        ),
        barmode='group'
    )
    
    return fig 