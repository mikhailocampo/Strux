import plotly.graph_objects as go
from typing import Dict, Any

def create_annotation_plot(field_data: Dict[str, Any]) -> go.Figure:
    """Create comparison plot between predictions and annotations."""
    fig = go.Figure()
    
    predictions = field_data.get("predictions", [])
    annotations = field_data.get("annotations", [])
    
    if not predictions:  # If no predictions, return empty plot
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=list(range(len(predictions))),
        y=predictions,
        name="Predictions",
        mode="lines+markers"
    ))
    
    # Add annotation line if available
    if annotations:
        fig.add_trace(go.Scatter(
            x=list(range(len(annotations))),
            y=annotations,
            name="Ground Truth",
            mode="lines+markers"
        ))
    
    fig.update_layout(
        title=f"{field_data['name']} Comparison",
        xaxis_title="Sample Index",
        yaxis_title="Value",
        showlegend=True
    )
    
    return fig

def create_experiment_plot(field_data: Dict[str, Any]) -> go.Figure:
    """Create comparison plot between experiment runs."""
    fig = go.Figure()
    
    current_values = field_data.get("current", [])
    baseline_values = field_data.get("baseline", [])
    
    if not current_values:  # If no current values, return empty plot
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    # Add current run line
    fig.add_trace(go.Scatter(
        x=list(range(len(current_values))),
        y=current_values,
        name="Current Run",
        mode="lines+markers"
    ))
    
    # Add baseline line if available
    if baseline_values:
        fig.add_trace(go.Scatter(
            x=list(range(len(baseline_values))),
            y=baseline_values,
            name="Baseline",
            mode="lines+markers"
        ))
    
    fig.update_layout(
        title=f"{field_data['name']} Comparison",
        xaxis_title="Sample Index",
        yaxis_title="Value",
        showlegend=True
    )
    
    return fig 