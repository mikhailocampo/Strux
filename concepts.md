# Core Concepts

Strux is built around a few key concepts that work together to enable structured output validation and regression testing. Understanding these core concepts will help you effectively use the framework.

## 1. Schemas

At the heart of Strux are Pydantic models that define the structure of your inputs and outputs. These schemas:
- Enforce type safety
- Provide clear documentation
- Enable automatic validation

Example schema definition:
```python
from pydantic import BaseModel
from typing import Literal

class Review(BaseModel):
    text: str
class ReviewAnalysis(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral']
    rating: float
```


## 2. Data Sources

Data sources provide a standardized way to load your test data. Strux includes built-in support for:

- **PostgresDataSource**: Load data directly from PostgreSQL tables/queries
- **CSVDataSource**: Load data from CSV files
- Custom sources: Extend the `DataSource` base class for your needs

Example PostgreSQL data source (referencing `cookbook/basic.py`):
```python
from strux import PostgresDataSource

data_source = PostgresDataSource(
    connection_string="postgresql://user:password@localhost:5432/mydatabase",
    query="SELECT * FROM reviews"
)
```


## 3. Validation Strategies

Strategies define how field values should be compared. Strux provides several built-in strategies:

- **exact_match()**: Values must be exactly equal
- **absolute_deviation(tolerance)**: Numeric values within a fixed tolerance
- **relative_deviation(percentage)**: Numeric values within a percentage
- **subset()**: One value must be a subset of another

Example configuration (referencing `cookbook/multi-experiments.ipynb`):
```python
config = RegressionConfig(
    target_schema=ReviewAnalysis
)
config.configure_field("sentiment", strategy=exact_match(), threshold=0.8)
config.configure_field("rating", strategy=absolute_deviation(0.5), threshold=0.9)
```


## 4. Pipeline Steps

A pipeline step is a function that:
1. Takes a single input record
2. Performs some computation or model inference
3. Returns a structured output matching your schema

Steps can be chained together to form more complex pipelines. Each step's output is automatically validated against its schema.

Example step:
```python
def analyze_review(data: Review) -> ReviewAnalysis:
    # Your model inference logic here
    text = data.text.lower()
    if "love" in text:
        return ReviewAnalysis(sentiment="positive", rating=5.0)
    elif "terrible" in text:
        return ReviewAnalysis(sentiment="negative", rating=1.0)
    else:
        return ReviewAnalysis(sentiment="neutral", rating=3.0)
```


## 5. Regression Config

The `RegressionConfig` class defines validation rules for your pipeline:

- Which fields to validate
- What strategies to use for comparison
- Thresholds for passing/failing
- Whether to use strict or relaxed validation

Example configuration:
```python
config = RegressionConfig(
    target_schema=ReviewAnalysis
)
```

### Configuring individual field validation

You can configure individual field validation rules using the `configure_field` method. This allows you to specify different strategies and thresholds for each field.

Example:
```python
config.configure_field("sentiment", strategy=exact_match(), threshold=0.8)
config.configure_field("rating", strategy=absolute_deviation(0.5), threshold=0.9)
```


## 6. Results and Reports

Strux provides rich output formats to help you understand your model's performance:

### Baseline Creation
The first run of your pipeline can be saved as a baseline:
```python
results = pipeline.run()
results.to_baseline("baseline.json")
```

### Regression Testing
Subsequent runs can be compared against the baseline:
```python
results = pipeline.run(baseline_path="baseline.json")
```


### HTML Reports
Strux generates interactive HTML reports showing:
- Overall pass/fail status
- Per-field metrics
- Sample-level comparisons
- Visualizations of changes

The reports highlight:
- 🟢 Improvements
- 🔴 Regressions
- ⚪ Unchanged results

```python
results.to_html("comparison_report.html")
```


## Best Practices

1. **Schema Design**
   - Make fields as specific as possible using Literal types
   - Use Pydantic validators for complex validation
   - Document expected value ranges and constraints

2. **Validation Strategy Selection**
   - Use `exact_match()` for categorical fields
   - Use `absolute_deviation()` for bounded numeric values
   - Use `relative_deviation()` for unbounded numeric values
   - Use `subset()` for list-like fields where the order of elements is not important

3. **Pipeline Organization**
   - Keep steps focused and single-purpose
   - Use meaningful step names
   - Consider breaking complex pipelines into multiple steps

4. **Version Control**
   - Store baselines in version control
   - Include baseline metadata (model version, date, etc.)
   - Document significant changes between versions
