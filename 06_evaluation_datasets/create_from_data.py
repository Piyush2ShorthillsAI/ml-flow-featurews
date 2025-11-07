"""
Example: Create Evaluation Datasets from Data
Demonstrates creating datasets from dictionaries and DataFrames
"""
import sys
sys.path.append('..')

import mlflow
from config import Config
from mlflow.genai.datasets import create_dataset
import pandas as pd


def main():
    print("\n" + "="*70)
    print("Evaluation Datasets - Create from Data")
    print("="*70 + "\n")
    
    Config.setup_mlflow()
    experiment = mlflow.get_experiment_by_name(Config.MLFLOW_EXPERIMENT_NAME)
    
    print("1️⃣  Creating dataset from dictionaries...")
    
    # Create dataset
    dataset = create_dataset(
        name="qa_evaluation_dataset",
        experiment_id=experiment.experiment_id,
        tags={"type": "qa", "version": "v1"}
    )
    print(f"✅ Created dataset: {dataset.name}\n")
    
    # Add records from dictionaries
    print("2️⃣  Adding records...")
    
    records = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {
                "answer": "MLflow is an open-source platform",
                "key_concepts": ["platform", "ML", "lifecycle"]
            }
        },
        {
            "inputs": {"question": "What is machine learning?"},
            "expectations": {
                "answer": "ML is a subset of AI",
                "key_concepts": ["AI", "data", "learning"]
            }
        }
    ]
    
    dataset.merge_records(records)
    print(f"✅ Added {len(records)} records\n")
    
    # Add from DataFrame
    print("3️⃣  Adding records from DataFrame...")
    
    df_data = pd.DataFrame({
        "inputs.question": ["What is Python?", "What is deep learning?"],
        "expectations.answer": ["Python is a programming language", "Deep learning uses neural networks"],
    })
    
    # Convert to record format
    df_records = []
    for _, row in df_data.iterrows():
        record = {
            "inputs": {"question": row["inputs.question"]},
            "expectations": {"answer": row["expectations.answer"]}
        }
        df_records.append(record)
    
    dataset.merge_records(df_records)
    print(f"✅ Added {len(df_records)} records from DataFrame\n")
    
    # View dataset info
    print("4️⃣  Dataset information...")
    print(f"   Name: {dataset.name}")
    print(f"   Tags: {dataset.tags}")
    print(f"   Total records: {len(records) + len(df_records)}\n")
    
    print("="*70)
    print("✨ Evaluation datasets provide:")
    print("   • Centralized test management")
    print("   • Version control for test data")
    print("   • Reusable evaluation suites")
    print(f"\n💡 Use dataset in evaluation with mlflow.genai.evaluate()")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

