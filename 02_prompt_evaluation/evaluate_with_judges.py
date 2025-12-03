"""
Example: Evaluate Prompts with LLM Judges
Demonstrates using LLM-as-a-Judge scorers with make_judge API
"""
import sys
sys.path.append('..')

import mlflow
from config import Config
import google.generativeai as genai
from utils import create_sample_qa_data, print_evaluation_results

# Import judge creation
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Guidelines, Correctness

# Enable Gemini tracing
mlflow.gemini.autolog()


def main():
    """Evaluate prompts using LLM judges"""
    
    print("\n" + "="*70)
    print("Prompt Evaluation - LLM Judges")
    print("="*70 + "\n")
    
    # Validate config
    if not Config.GEMINI_API_KEY:
        print("⚠️  ERROR: GEMINI_API_KEY not set")
        return
    
    # Setup
    Config.setup_mlflow()
    genai.configure(api_key=Config.GEMINI_API_KEY)
    
    # ============================================================
    # 1. Create Prediction Function
    # ============================================================
    print("1️⃣  Setting up prediction function...")
    
    @mlflow.trace
    def predict_fn(question: str) -> str:
        try:
            prompt = mlflow.genai.load_prompt("prompts:/qa_prompt_chat@latest")
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
                generation_config=genai.types.GenerationConfig(temperature=0.7)
            )
            return response.text
        except Exception as e:
            return f"Error: {e}"
    
    print("✅ Prediction function ready\n")
    
    # ============================================================
    # 2. Create LLM Judge Scorers
    # ============================================================
    print("2️⃣  Creating LLM judge scorers...")
    
    # Judge 1: Quality Assessment
    quality_judge = make_judge(
        name="response_quality",
        instructions=(
            "Evaluate the quality of {{ outputs }} in response to {{ inputs }}.\n"
            "Consider:\n"
            "- Accuracy of information\n"
            "- Completeness of answer\n"
            "- Clarity of explanation\n\n"
            "Rate as 'excellent', 'good', 'fair', or 'poor'."
        ),
        model="gemini:/gemini-2.0-flash-001",
    )
    print("✅ Created quality judge")
    
    # Judge 2: Technical Accuracy
    technical_judge = make_judge(
        name="technical_accuracy",
        instructions=(
            "Evaluate the technical accuracy of {{ outputs }} for the question in {{ inputs }}.\n"
            "Check if the response:\n"
            "- Contains correct technical information\n"
            "- Uses appropriate terminology\n"
            "- Avoids misconceptions\n\n"
            "Rate as 'accurate', 'mostly_accurate', 'partially_accurate', or 'inaccurate'."
        ),
        model="gemini:/gemini-2.0-flash-001",
    )
    print("✅ Created technical accuracy judge")
    
    # Judge 3: Helpfulness
    helpfulness_judge = make_judge(
        name="helpfulness",
        instructions=(
            "Evaluate how helpful {{ outputs }} is for answering {{ inputs }}.\n"
            "Consider:\n"
            "- Directly addresses the question\n"
            "- Provides actionable information\n"
            "- Includes relevant examples if appropriate\n\n"
            "Rate as 'very_helpful', 'helpful', 'somewhat_helpful', or 'not_helpful'."
        ),
        model="gemini:/gemini-2.0-flash-001",
    )
    print("✅ Created helpfulness judge")
    
    # Judge 4: Expected Content Match (with expectations)
    content_match_judge = make_judge(
        name="expected_content",
        instructions=(
            "Compare {{ outputs }} with the expected response in {{ expectations }}.\n"
            "Check if the output covers the same key concepts even if worded differently.\n\n"
            "Rate as 'matches', 'mostly_matches', 'partially_matches', or 'no_match'."
        ),
        model="gemini:/gemini-2.0-flash-001",
    )
    print("✅ Created content match judge")
    
    # Add built-in scorers
    correctness = Correctness(name="correctness")
    professional = Guidelines(
        name="professional",
        guidelines="Response must be professional and appropriate for a knowledge assistant"
    )
    
    print("✅ Added built-in scorers\n")
    
    scorers = [
        quality_judge,
        technical_judge,
        helpfulness_judge,
        content_match_judge,
        correctness,
        professional
    ]
    
    # ============================================================
    # 3. Prepare Dataset
    # ============================================================
    print("3️⃣  Preparing evaluation dataset...")
    
    eval_data = create_sample_qa_data()
    print(f"✅ Dataset ready: {len(eval_data)} samples\n")
    
    # ============================================================
    # 4. Run Evaluation with Judges
    # ============================================================
    print("4️⃣  Running evaluation with LLM judges...")
    print("   ⏳ This will take longer as judges are LLM-based...\n")
    
    with mlflow.start_run(run_name="prompt_eval_with_judges"):
        mlflow.log_param("prompt", "qa_prompt_chat@latest")
        mlflow.log_param("judge_model", "gemini-2.0-flash-001")
        mlflow.log_param("num_judges", len(scorers))
        mlflow.set_tag("evaluation_type", "llm_judge")
        
        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=scorers
        )
        
        print("✅ Evaluation complete!\n")
        
        # ============================================================
        # 5. Analyze Results
        # ============================================================
        print_evaluation_results(results, "LLM Judge Evaluation Results")
        
        # Show judge-specific insights
        if hasattr(results, 'tables') and 'eval_results_table' in results.tables:
            df = results.tables['eval_results_table']
            
            print("🤖 LLM Judge Insights:\n")
            
            # Analyze quality ratings
            if 'response_quality/score' in df.columns:
                quality_dist = df['response_quality/score'].value_counts()
                print(f"   Quality Distribution:")
                for rating, count in quality_dist.items():
                    print(f"      {rating}: {count} samples")
            
            # Analyze technical accuracy
            if 'technical_accuracy/score' in df.columns:
                accuracy_dist = df['technical_accuracy/score'].value_counts()
                print(f"\n   Technical Accuracy Distribution:")
                for rating, count in accuracy_dist.items():
                    print(f"      {rating}: {count} samples")
            
            # Check rationales (if available)
            if 'response_quality/rationale' in df.columns:
                print(f"\n   Sample Rationales:")
                for idx in range(min(2, len(df))):
                    if pd.notna(df.iloc[idx]['response_quality/rationale']):
                        print(f"      Sample {idx+1}: {df.iloc[idx]['response_quality/rationale'][:100]}...")
            
            print()
    
    # ============================================================
    # Summary
    # ============================================================
    print("="*70)
    print("✨ Summary")
    print("="*70)
    print(f"✅ Evaluated with {len(scorers)} scorers (4 LLM judges + 2 built-in)")
    print(f"✅ Processed {len(eval_data)} samples")
    print(f"✅ LLM judges provide detailed rationales for scores")
    print(f"\n💡 Benefits of LLM Judges:")
    print(f"   • More nuanced evaluation than simple metrics")
    print(f"   • Natural language rationales for debugging")
    print(f"   • Can evaluate subjective qualities")
    print(f"   • Flexible instructions for custom criteria")
    print(f"\n💡 View results in MLflow UI: {Config.MLFLOW_TRACKING_URI}")
    print("="*70 + "\n")


if __name__ == "__main__":
    import pandas as pd
    main()

