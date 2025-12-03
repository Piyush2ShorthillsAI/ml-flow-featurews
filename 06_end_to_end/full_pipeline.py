"""
Example: End-to-End MLflow Pipeline
Complete workflow demonstrating all major features together
"""
import sys
sys.path.append('..')

import mlflow
from config import Config
import google.generativeai as genai
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness
from mlflow.entities import Feedback

# Enable Gemini tracing
mlflow.gemini.autolog()


def main():
    print("\n" + "="*70)
    print("End-to-End MLflow Pipeline")
    print("="*70 + "\n")
    
    if not Config.GEMINI_API_KEY:
        print("⚠️  ERROR: GEMINI_API_KEY not set")
        return
    
    Config.setup_mlflow()
    genai.configure(api_key=Config.GEMINI_API_KEY)
    
    # Step 1: Register Prompt
    print("1️⃣  Registering prompt...")
    try:
        prompt = mlflow.genai.register_prompt(
            name="e2e_qa_prompt",
            template="Answer concisely: {{question}}",
            commit_message="E2E demo prompt"
        )
        print(f"✅ Prompt registered: v{prompt.version}\n")
    except Exception as e:
        print(f"⚠️  Prompt may already exist: {e}\n")
        prompt = mlflow.genai.load_prompt("prompts:/e2e_qa_prompt@latest")
    
    # Step 2: Create Prediction Function
    print("2️⃣  Setting up prediction...")
    
    @mlflow.trace
    def predict_fn(question: str) -> str:
        prompt_obj = mlflow.genai.load_prompt("prompts:/e2e_qa_prompt@latest")
        formatted = prompt_obj.format(question=question)
        
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        response = model.generate_content(formatted)
        return response.text
    
    print("✅ Prediction function ready\n")
    
    # Step 3: Create Evaluation Data
    print("3️⃣  Creating evaluation dataset...")
    
    eval_data = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is an ML platform"}
        },
        {
            "inputs": {"question": "What is AI?"},
            "expectations": {"answer": "AI is machine intelligence"}
        }
    ]
    print(f"✅ Dataset: {len(eval_data)} samples\n")
    
    # Step 4: Define Scorers
    print("4️⃣  Creating scorers...")
    
    @scorer
    def is_concise(outputs: str) -> Feedback:
        words = len(outputs.split())
        score = 1.0 if words <= 50 else 0.5
        return Feedback(value=score, rationale=f"{words} words")
    
    scorers = [Correctness(), is_concise]
    print(f"✅ {len(scorers)} scorers ready\n")
    
    # Step 5: Run Evaluation
    print("5️⃣  Running evaluation...")
    
    with mlflow.start_run(run_name="e2e_pipeline"):
        # Log metadata
        mlflow.log_param("prompt_version", prompt.version)
        mlflow.log_param("model", "gemini-2.0-flash-001")
        mlflow.set_tag("pipeline_type", "end_to_end")
        
        # Evaluate
        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=scorers
        )
        
        print(f"✅ Evaluation complete")
        print(f"   Metrics: {results.metrics}\n")
    
    # Step 6: Summary
    print("="*70)
    print("✨ End-to-End Pipeline Complete!")
    print("="*70)
    print("✅ Prompt registry ✓")
    print("✅ Tracing ✓")
    print("✅ Evaluation ✓")
    print("✅ Experiment tracking ✓")
    print(f"\n💡 View complete workflow in: {Config.MLFLOW_TRACKING_URI}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

