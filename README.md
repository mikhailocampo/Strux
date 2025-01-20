# Strux: A Python framework for structured output model versioning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch) ![Static Badge](https://img.shields.io/badge/Pydantic-red?logo=pydantic&label=%20&labelColor=gray&color=%23E92063&link=https%3A%2F%2Fpypi.org%2Fproject%2Fpydantic%2F)

Strux is a Python framework designed to make comparing and validating model outputs across multiple experiments simple and straightforward. Its primary goal is to enable fast iteration during model development by making it extremely easy to:
- Compare an updated model’s outputs against a previously established baseline
- Track improvements and regressions at both the field and sample level
- Generate clear, automated reports to guide data scientists and engineers in calibrating or refining their models

## The Problem

Testing AI models (especially those generating structured outputs) requires more than just raw accuracy checks. As models become increasingly complex (e.g., multi-step pipelines, large language model inference with structured outputs), practitioners need:
1. Type-safe scrutiny of outputs against a schema.
2. Fine-grained inspection of changes in performance or behavior, even at the field level (e.g., sentiment vs. rating).
3. A way to confidently iterate and measure improvements without fear of accidentally introducing regressions.

Traditional testing frameworks don’t address these issues in a developer-friendly, integrated manner. They either require lots of custom scripts or rely on ad-hoc comparisons that can be error-prone and slow.


## Why Strux?

1. **Type-Safe with Pydantic**: Strux is built on Pydantic for schema validation, ensuring your model’s outputs conform to the expected structures. This catches errors early and eliminates ambiguity in your data pipeline.

2. **Simple Pipeline Abstractions**: Model inference is packaged into “steps” that can be chained. Strux automatically validates outputs after each step according to configurable thresholds.

3. **Comparison & Regression Testing**: Quickly see if your new model version outperforms or regresses vs. the baseline. Each field can have its own strategy (e.g., exact match, allowable numeric deviation).

4. **Rich Reporting**: Strux can generate HTML reports that visually highlight improvements, regressions, or unchanged results. Summaries and per-sample breakdowns give you the context you need to iterate quickly.

5. **Cookbook & Extensible**: Built-in data sources (PostgreSQL, CSV, etc.) help bootstrap your pipeline. Community and team contributions can extend Strux to fit your own data retrieval or custom validation needs.


## Getting Started

Below is a minimal example illustrating how to set up a Strux pipeline and perform a comparison against a baseline.

### Install Strux
```bash
pip install strux
```

### Step 1: Define Your Schemas

Use Pydantic models for both your input and output schemas:

```python
from pydantic import BaseModel
from typing import Literal

class Review(BaseModel):
    text: str
class ReviewAnalysis(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral']
    rating: float
```


### Step 2: Implement an Inference Function

This function takes a single input record (Review) and returns structured outputs (ReviewAnalysis):

```python
def analyze_review_v1(data: Review) -> ReviewAnalysis:
# Some simple logic or call a model endpoint
# Example: short, naive approach
text = data.text.lower()
if "love" in text:
sentiment = "positive"
rating = 5.0
elif "terrible" in text:
sentiment = "negative"
rating = 1.0
else:
sentiment = "neutral"
rating = 3.0
return ReviewAnalysis(sentiment=sentiment, rating=rating)
```


### Step 3: Build and Run a Pipeline

Strux pipelines tie together data loading, inference steps, and validation rules:

```python
from strux import CSVDataSource, RegressionConfig, Sequential, exact_match, absolute_deviation

# Load data from a CSV or other data source
data_source = CSVDataSource(
    file_path="reviews.csv", # CSV file with a "text" column
    schema=Review
)

# Configure validation rules for each field
config = RegressionConfig(
    target_schema=ReviewAnalysis
)
config.configure_field(
    "sentiment", 
    strategy=exact_match(), # Exact match strategy where the value must be exactly the same as the baseline
    threshold=0.8 # 80% of the samples must match the baseline or annotation
)
config.configure_field(
    "rating", 
    strategy=absolute_deviation(0.5), # Allowable value-based deviation of 0.5
    threshold=0.9 # 90% of the samples must match the baseline or annotation
)

# Build the pipeline
pipeline = Sequential.from_steps(
    data_source=data_source,
    steps=[("analyze_review_v1", analyze_review_v1, ReviewAnalysis)],
    config=config
)

# First run to create a baseline
results_v1 = pipeline.run()
results_v1.save_as_baseline("baselines/review_baseline.json")

# Compare against the baseline with a v2
def analyze_review_v2(data: Review) -> ReviewAnalysis:
    # A new and improved approach!
    # For demonstration, let’s just tweak thresholds slightly
    text = data.text.lower()
    if "love" in text or "amazing" in text:
        sentiment = "positive"
        rating = 5.0
    elif "terrible" in text or "broke" in text:
        sentiment = "negative"
        rating = 1.0
    else:
        sentiment = "neutral"
        rating = 3.0
    return ReviewAnalysis(sentiment=sentiment, rating=rating)
    
pipeline_v2 = Sequential.from_steps(
    data_source=data_source,
    steps=[
        ("analyze_review_v2", analyze_review_v2, ReviewAnalysis)
    ],
    config=config
)

# Run with baseline comparison
results_v2 = pipeline_v2.run(baseline_path="baselines/review_baseline.json")
print(results_v2.format_summary()) # Quick console summary

# Generate visual HTML report
results_v2.to_html("comparison_report.html")
```


You’ll see improvements, regressions, and unchanged samples clearly highlighted in both console output and the generated HTML file.

## Documentation

For detailed documentation, see the [docs/](docs/) directory:

- [Getting Started](docs/getting-started.md)
- [Core Concepts](docs/core-concepts.md)
- [API Reference](docs/api-reference.md)
- [Examples](docs/examples.md)

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

