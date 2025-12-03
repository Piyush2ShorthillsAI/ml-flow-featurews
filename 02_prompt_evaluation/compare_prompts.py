"""
Example: Compare Multiple Prompt Versions
Demonstrates how to compare different prompt versions side-by-side
"""
import sys
sys.path.append('..')

import mlflow
from config import Config
import google.generativeai as genai
from utils import create_sample_qa_data, compare_results

# Import scorers
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines
from mlflow.entities import Feedback

# Enable Gemini tracing
mlflow.gemini.autolog()


def main():
    """Compare multiple prompt versions"""
    
    print("\n" + "="*70)
    print("Prompt Evaluation - Compare Multiple Versions")
    print("="*70 + "\n")
    
    # Validate config
    if not Config.GEMINI_API_KEY:
        print("⚠️  ERROR: GEMINI_API_KEY not set")
        return
    
    # Setup
    Config.setup_mlflow()
    genai.configure(api_key=Config.GEMINI_API_KEY)
    
    # ============================================================
    # 1. Create Prediction Functions for Different Prompts
    # ============================================================
    print("1️⃣  Setting up prediction functions for different prompts...")
    
    def create_predict_fn(prompt_identifier: str):
        """Factory function to create prediction functions"""
        @mlflow.trace
        def predict(question: str) -> str:
            try:
                prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_identifier}")
                formatted = prompt.format(question=question)
                
                # Call Gemini
                model = genai.GenerativeModel('gemini-2.0-flash-001')
                
                # Convert chat messages to Gemini format
                if isinstance(formatted, list):
                    system_msg = ""
                    user_msg = ""
                    for msg in formatted:
                        if msg.get("role") == "system":
                            system_msg = msg.get("content", "")
                        elif msg.get("role") == "user":
                            user_msg = msg.get("content", "")
                    
                    full_prompt = f"{system_msg}\n\n{user_msg}" if system_msg else user_msg
                else:
                    full_prompt = formatted
                
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=150
                    )
                )
                return response.text
            except Exception as e:
                return f"Error: {e}"
        
        return predict
    
    # Define prompt versions to compare
    prompt_versions = {
        "Simple Prompt": "qa_prompt_simple@latest",
        "Chat V1": "qa_prompt_chat/1",
        "Chat V2 (Latest)": "qa_prompt_chat@latest",
    }
    
    print(f"✅ Will compare {len(prompt_versions)} prompt versions\n")
    
    # ============================================================
    # 2. Prepare Dataset and Scorers
    # ============================================================
    print("2️⃣  Preparing dataset and scorers...")
    
    eval_data = create_sample_qa_data()
    
    @scorer
    def clarity_score(outputs: str) -> Feedback:
        """Score based on clarity (simple heuristic)"""
        # Check for clear structure
        has_examples = any(word in outputs.lower() for word in ['example', 'for instance', 'such as'])
        has_clear_answer = len(outputs.split('.')) >= 2
        
        score = 0.5
        factors = []
        
        if has_examples:
            score += 0.25
            factors.append("includes examples")
        if has_clear_answer:
            score += 0.25
            factors.append("clear structure")
        
        return Feedback(
            value=min(score, 1.0),
            rationale=f"Clarity factors: {', '.join(factors) if factors else 'basic response'}"
        )
    
    scorers = [
        Correctness(name="correctness"),
        Guidelines(name="helpful", guidelines="The response should be helpful and informative"),
        clarity_score
    ]
    
    print(f"✅ Dataset: {len(eval_data)} samples")
    print(f"✅ Scorers: {len(scorers)} metrics\n")
    
    # ============================================================
    # 3. Evaluate Each Prompt Version
    # ============================================================
    print("3️⃣  Evaluating each prompt version...")
    print("   This will take a few moments...\n")
    
    all_results = {}
    
    for prompt_name, prompt_id in prompt_versions.items():
        print(f"   Evaluating: {prompt_name}...")
        
        try:
            predict_fn = create_predict_fn(prompt_id)
            
            with mlflow.start_run(run_name=f"eval_{prompt_name.replace(' ', '_')}"):
                mlflow.log_param("prompt_id", prompt_id)
                mlflow.log_param("prompt_name", prompt_name)
                mlflow.set_tag("evaluation_type", "prompt_comparison")
                
                results = mlflow.genai.evaluate(
                    data=eval_data,
                    predict_fn=predict_fn,
                    scorers=scorers
                )
                
                all_results[prompt_name] = results
                print(f"   ✅ {prompt_name} complete")
        
        except Exception as e:
            print(f"   ⚠️  Error evaluating {prompt_name}: {e}")
    
    print()
    
    # ============================================================
    # 4. Compare Results
    # ============================================================
    print("4️⃣  Comparing results across prompt versions...\n")
    
    compare_results(all_results)
    
    # Detailed comparison
    print("📊 Detailed Comparison:\n")
    
    comparison_metrics = {}
    for prompt_name, results in all_results.items():
        if hasattr(results, 'metrics'):
            comparison_metrics[prompt_name] = results.metrics
    
    if comparison_metrics:
        # Find best performing prompt for each metric
        print("🏆 Best Performing Prompts:\n")
        
        all_metric_names = set()
        for metrics in comparison_metrics.values():
            all_metric_names.update(metrics.keys())
        
        for metric_name in sorted(all_metric_names):
            best_score = -float('inf')
            best_prompt = None
            
            for prompt_name, metrics in comparison_metrics.items():
                if metric_name in metrics:
                    score = metrics[metric_name]
                    if isinstance(score, (int, float)) and score > best_score:
                        best_score = score
                        best_prompt = prompt_name
            
            if best_prompt:
                print(f"   {metric_name}: {best_prompt} ({best_score:.4f})")
    
    print()
    
    # ============================================================
    # 5. Recommendations
    # ============================================================
    print("="*70)
    print("💡 Recommendations")
    print("="*70)
    
    if comparison_metrics:
        # Calculate average score for each prompt
        avg_scores = {}
        for prompt_name, metrics in comparison_metrics.items():
            numeric_metrics = [v for v in metrics.values() if isinstance(v, (int, float))]
            if numeric_metrics:
                avg_scores[prompt_name] = sum(numeric_metrics) / len(numeric_metrics)
        
        if avg_scores:
            best_overall = max(avg_scores.items(), key=lambda x: x[1])
            print(f"\n🥇 Overall Best Prompt: {best_overall[0]}")
            print(f"   Average Score: {best_overall[1]:.4f}")
            print(f"\n✅ Consider promoting this version to production")
    
    print(f"\n💡 View detailed comparison in MLflow UI: {Config.MLFLOW_TRACKING_URI}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

