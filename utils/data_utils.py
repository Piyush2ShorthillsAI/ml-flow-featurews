"""
Data utility functions for creating sample datasets
"""
import pandas as pd
from typing import List, Dict, Any


def create_sample_qa_data() -> List[Dict[str, Any]]:
    """Create sample Q&A evaluation dataset"""
    return [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {
                "expected_response": "MLflow is an open-source platform for managing the machine learning lifecycle",
                "key_concepts": ["open-source", "ML lifecycle", "tracking"]
            }
        },
        {
            "inputs": {"question": "What is machine learning?"},
            "expectations": {
                "expected_response": "Machine learning is a subset of AI that enables systems to learn from data",
                "key_concepts": ["AI", "data", "learning", "algorithms"]
            }
        },
        {
            "inputs": {"question": "Explain what is prompt engineering?"},
            "expectations": {
                "expected_response": "Prompt engineering is the practice of designing effective prompts for LLMs",
                "key_concepts": ["LLM", "prompts", "design", "optimization"]
            }
        },
        {
            "inputs": {"question": "What are the benefits of model versioning?"},
            "expectations": {
                "expected_response": "Model versioning helps track changes, enables rollback, and ensures reproducibility",
                "key_concepts": ["versioning", "tracking", "reproducibility", "rollback"]
            }
        },
        {
            "inputs": {"question": "How does evaluation help in ML?"},
            "expectations": {
                "expected_response": "Evaluation helps measure model performance and guide improvements",
                "key_concepts": ["performance", "metrics", "improvement", "quality"]
            }
        }
    ]


def create_sample_chat_data() -> List[Dict[str, Any]]:
    """Create sample chat/conversation data"""
    return [
        {
            "inputs": {
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ]
            },
            "expectations": {"answer": "Paris"}
        },
        {
            "inputs": {
                "messages": [
                    {"role": "user", "content": "What is 2 + 2?"}
                ]
            },
            "expectations": {"answer": "4"}
        },
        {
            "inputs": {
                "messages": [
                    {"role": "user", "content": "Who wrote Romeo and Juliet?"}
                ]
            },
            "expectations": {"answer": "William Shakespeare"}
        }
    ]


def create_summarization_data() -> List[Dict[str, Any]]:
    """Create sample summarization dataset"""
    return [
        {
            "inputs": {
                "text": "Artificial intelligence has transformed how businesses operate in the 21st century. "
                        "Companies are leveraging AI for everything from customer service to supply chain optimization."
            },
            "expectations": {
                "summary": "AI has revolutionized business operations through automation and optimization."
            }
        },
        {
            "inputs": {
                "text": "Climate change continues to affect ecosystems worldwide at an alarming rate. "
                        "Rising temperatures lead to extreme weather events and threaten coastal communities."
            },
            "expectations": {
                "summary": "Climate change is causing environmental damage through extreme weather."
            }
        }
    ]


def data_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert evaluation data to pandas DataFrame"""
    rows = []
    for item in data:
        row = {}
        # Flatten inputs
        for key, value in item.get("inputs", {}).items():
            row[f"inputs.{key}"] = value
        # Flatten expectations
        for key, value in item.get("expectations", {}).items():
            row[f"expectations.{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def print_dataset_info(data: List[Dict[str, Any]], name: str = "Dataset"):
    """Print information about a dataset"""
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}")
    print(f"Total samples: {len(data)}")
    if data:
        print(f"\nSample record:")
        print(f"  Inputs: {data[0].get('inputs', {})}")
        print(f"  Expectations: {data[0].get('expectations', {})}")
    print(f"{'='*60}\n")

