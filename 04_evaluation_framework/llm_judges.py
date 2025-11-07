"""
Example: LLM-as-a-Judge Scorers
Using Guidelines and make_judge APIs
"""
import sys
sys.path.append('..')

import mlflow
from config import Config
from openai import OpenAI
from utils import create_sample_qa_data
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Guidelines, Correctness


def main():
    print("\n" + "="*70)
    print("LLM-as-a-Judge Scorers")
    print("="*70 + "\n")
    
    if not Config.OPENAI_API_KEY:
        print("⚠️  ERROR: OPENAI_API_KEY not set")
        return
    
    Config.setup_mlflow()
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    # Prediction function
    @mlflow.trace
    def predict_fn(question: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content
    
    # LLM Judge Scorers
    print("Creating LLM judges...")
    
    quality_judge = make_judge(
        name="quality",
        instructions="Rate {{ outputs }} quality for {{ inputs }} as excellent/good/poor",
        model="openai:/gpt-4o-mini"
    )
    
    completeness_judge = make_judge(
        name="completeness",
        instructions="Is {{ outputs }} complete for {{ inputs }}? Rate: complete/partial/incomplete",
        model="openai:/gpt-4o-mini"
    )
    
    guidelines_scorer = Guidelines(
        name="professional",
        guidelines="Response must be professional and clear"
    )
    
    correctness_scorer = Correctness(name="correctness")
    
    scorers = [quality_judge, completeness_judge, guidelines_scorer, correctness_scorer]
    print(f"✅ Created {len(scorers)} LLM judges\n")
    
    # Evaluate
    print("Running evaluation...")
    eval_data = create_sample_qa_data()[:2]  # Use 2 samples for speed
    
    with mlflow.start_run(run_name="llm_judges"):
        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=scorers
        )
        print("✅ Evaluation complete!")
        print(f"\nMetrics: {results.metrics}\n")
    
    print("="*70)
    print("💡 LLM judges provide nuanced, human-like evaluation")
    print(f"💡 View in MLflow UI: {Config.MLFLOW_TRACKING_URI}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

