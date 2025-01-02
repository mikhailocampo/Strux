from typing import List, Literal

import pandas as pd
from strux import PostgresDataSource, RegressionConfig, Sequential
from pydantic import BaseModel

from strux.strategies import exact_match


# Define input/output schemas
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class InputSchema(BaseModel):
    chat_history: List[ChatMessage]

class OutputSchema(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]

# Define inference function
def inference(input: InputSchema) -> OutputSchema:
    return OutputSchema(sentiment="positive")

connection_params={
        "user": "YOUR_USER",
        "password": "YOUR_PASSWORD",
        "host": "YOUR_HOST",
        "port": 6543,
        "database": "YOUR_DATABASE"
    }

def setup_pipeline(connection_params: dict) -> Sequential:
    source = PostgresDataSource(
        query="SELECT * FROM matador_s_idx LIMIT 1",
        connection_params=connection_params,
        schema=InputSchema,
        json_columns={"chat_history": List[ChatMessage]}
    )

    config = RegressionConfig(
        target_schema=OutputSchema,
        strict_fields=["sentiment"],
    )
    
    config.configure_field("sentiment", strategy=exact_match())
    
    return Sequential.from_steps(
        data_source=source,
        steps=[
            ("sentiment", inference, OutputSchema)
        ],
        config=config
    )

def main():
    pipeline = setup_pipeline(connection_params)
    
    # First run - create baseline
    results = pipeline.run(
        baseline_path="baselines/sentiment_baseline.json"
    )
    
    # Check if this is first run (no baseline)
    is_first_run = all(
        "is_first_run" in step.metadata and step.metadata["is_first_run"]
        for step in results.step_validations
    )
    
    if is_first_run:
        print("\nFirst run completed. Would you like to save as baseline? (y/n)")
        if input().lower() == 'y':
            results.save_as_baseline("baselines/sentiment_baseline.json")
            print("\nNext steps:")
            print("1. Make changes to your model")
            print("2. Run regression test against baseline:")
            print("   pipeline.run(baseline_path='baselines/sentiment_baseline.json')")
    else:
        if results.passed:
            print("\nAll tests passed! Your changes maintain baseline behavior.")
        else:
            print("\nRegression test failed. Review the differences:")
            for step in results.get_failed_steps():
                print(f"\n{step.format_summary()}")

if __name__ == "__main__":
    main()