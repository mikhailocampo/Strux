# Strux

Strux is a Python library for building type-safe regression testing pipelines for structured outputs. It enables developers to easily validate model outputs against expected schemas and thresholds with minimal boilerplate.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Key Features

- 🔒 **Type-Safe**: Built on Pydantic for robust schema validation and type checking
- 🔄 **Flexible Pipelines**: Chain multiple inference steps with automatic schema validation
- 📊 **Configurable Validation**: Define strict, relaxed, or custom validation strategies
- 🔌 **Extensible**: Built-in PostgreSQL support with easy extension for other data sources
- 📈 **Rich Results**: Detailed validation reports with field-level insights

## Quick Start
```python
from strux import Sequential, RegressionConfig, ValidationLevel
from pydantic import BaseModel
```

### Define your Schemas
```python
class InputSchema(BaseModel):
    value: float
    text: str

class OutputSchema(BaseModel):
    processed: bool
    confidence: float
```

### Define your Inference Function
```python
def process_text(data: InputSchema) -> OutputSchema:
    return OutputSchema(
    processed=data.text.upper(),
    confidence=data.value 100
)
```

### Create and run pipeline
```python
pipeline = Sequential.from_steps(
    data_source=your_data_source,
    steps=[
        ("process", process_text, OutputSchema)
    ],
config=RegressionConfig(
    OutputSchema,
    strict_fields=["processed"],
    relaxed_fields=["confidence"]
    )
)
results = pipeline.run()
```

## Installation
```bash
pip install strux
```


## Why Strux?

- **Type Safety**: Catch schema mismatches early with Pydantic validation
- **Flexible Validation**: Configure different validation levels per field
- **Pipeline Composition**: Chain multiple inference steps with automatic validation
- **Production Ready**: Built-in support for batch processing and error handling

## Documentation

For detailed documentation, see the [docs/](docs/) directory:

- [Getting Started](docs/getting-started.md)
- [Core Concepts](docs/core-concepts.md)
- [API Reference](docs/api-reference.md)
- [Examples](docs/examples.md)

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

