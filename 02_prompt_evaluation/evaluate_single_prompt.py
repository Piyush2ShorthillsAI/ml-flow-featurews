"""
Example: Evaluate a Single Prompt
Demonstrates how to evaluate prompts using MLflow's evaluation framework
"""
import sys
sys.path.append('..')

import mlflow
from config import Config
from openai import OpenAI
from utils import create_sample_qa_data, print_evaluation_results

# Import scorers
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines
from mlflow.entities import Feedback


def main():
    """Evaluate a single prompt with custom and built-in scorers"""
    
    print("\n" + "="*70)
    print("Prompt Evaluation - Single Prompt Example")
    print("="*70 + "\n")
    
    # Validate config
    if not Config.OPENAI_API_KEY:
        print("⚠️  ERROR: OPENAI_API_KEY not set in .env file")
        print("Please set your OpenAI API key and try again")
        return
    
    # Setup MLflow
    Config.setup_mlflow()
    
    # Setup OpenAI client
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    # ============================================================
    # 1. Define Prediction Function with Prompt Registry
    # ============================================================
    print("1️⃣  Setting up prediction function...")
    
    prompt_name = "qa_prompt_chat"
    
    @mlflow.trace
    def predict_fn(question: str) -> str:
        """Predict function that uses registered prompt"""
        try:
            # Load prompt from registry
            prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@latest")
            formatted_messages = prompt.format(question=question)
            
            # Call OpenAI
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=formatted_messages,
                temperature=Config.DEFAULT_TEMPERATURE,
                max_tokens=Config.DEFAULT_MAX_TOKENS
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️  Error: {e}")
            print(f"   Make sure to run ../01_prompt_registry/register_prompts.py first")
            return "Error: Could not generate response"
    
    print(f"✅ Prediction function ready")
    print(f"   Using prompt: {prompt_name}@latest\n")
    
    # ============================================================
    # 2. Prepare Evaluation Dataset
    # ============================================================
    print("2️⃣  Preparing evaluation dataset...")
    
    eval_data = create_sample_qa_data()
    print(f"✅ Created dataset with {len(eval_data)} samples\n")
    
    # ============================================================
    # 3. Define Custom Scorers
    # ============================================================
    print("3️⃣  Defining evaluation scorers...")
    
    @scorer
    def is_concise(outputs: str) -> bool:
        """Check if response is concise (under 100 words)"""
        word_count = len(outputs.split())
        return word_count <= 100
    
    @scorer
    def contains_key_concepts(outputs: str, expectations: dict) -> Feedback:
        """Check if response contains expected key concepts"""
        key_concepts = expectations.get("key_concepts", [])
        if not key_concepts:
            return Feedback(value=1.0, rationale="No key concepts to check")
        
        output_lower = outputs.lower()
        found_concepts = [concept for concept in key_concepts if concept.lower() in output_lower]
        score = len(found_concepts) / len(key_concepts)
        
        return Feedback(
            value=score,
            rationale=f"Found {len(found_concepts)}/{len(key_concepts)} key concepts: {found_concepts}"
        )
    
    @scorer
    def response_length_score(outputs: str) -> Feedback:
        """Score based on response length"""
        length = len(outputs.split())
        
        # Optimal length between 20-80 words
        if 20 <= length <= 80:
            score = 1.0
            rationale = f"Optimal length: {length} words"
        elif length < 20:
            score = 0.5
            rationale = f"Too short: {length} words"
        else:
            score = 0.7
            rationale = f"A bit long: {length} words"
        
        return Feedback(value=score, rationale=rationale)
    
    # Built-in scorers
    correctness_scorer = Correctness(name="factual_correctness")
    professional_tone = Guidelines(
        name="professional_tone",
        guidelines="The response should be professional, clear, and well-structured"
    )
    
    scorers = [
        correctness_scorer,
        professional_tone,
        is_concise,
        contains_key_concepts,
        response_length_score
    ]
    
    print(f"✅ Defined {len(scorers)} scorers:")
    for sc in scorers:
        scorer_name = getattr(sc, 'name', sc.__name__ if hasattr(sc, '__name__') else str(sc))
        print(f"   - {scorer_name}")
    print()
    
    # ============================================================
    # 4. Run Evaluation
    # ============================================================
    print("4️⃣  Running evaluation...")
    print("   This may take a moment as it processes multiple samples...\n")
    
    with mlflow.start_run(run_name="prompt_evaluation_single"):
        # Log prompt info
        mlflow.log_param("prompt_name", prompt_name)
        mlflow.log_param("prompt_alias", "latest")
        mlflow.log_param("model", "gpt-4o-mini")
        mlflow.log_param("temperature", Config.DEFAULT_TEMPERATURE)
        
        # Run evaluation
        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=scorers
        )
        
        print("✅ Evaluation complete!\n")
        
        # ============================================================
        # 5. Display Results
        # ============================================================
        print_evaluation_results(results, "Prompt Evaluation Results")
        
        # Additional analysis
        if hasattr(results, 'tables') and 'eval_results_table' in results.tables:
            df = results.tables['eval_results_table']
            
            print("📊 Detailed Analysis:")
            print(f"   Total samples evaluated: {len(df)}")
            
            # Calculate pass rate for boolean scorers
            if 'is_concise/score' in df.columns:
                concise_rate = df['is_concise/score'].mean()
                print(f"   Concise responses: {concise_rate*100:.1f}%")
            
            if 'contains_key_concepts/score' in df.columns:
                avg_concept_coverage = df['contains_key_concepts/score'].mean()
                print(f"   Average concept coverage: {avg_concept_coverage*100:.1f}%")
            
            print()
    
    # ============================================================
    # Summary
    # ============================================================
    print("="*70)
    print("✨ Summary")
    print("="*70)
    print(f"✅ Evaluated prompt: {prompt_name}@latest")
    print(f"✅ Processed {len(eval_data)} test cases")
    print(f"✅ Used {len(scorers)} evaluation metrics")
    print(f"✅ Results logged to MLflow")
    print(f"\n💡 View detailed results in MLflow UI: {Config.MLFLOW_TRACKING_URI}")
    print("💡 Next: Compare multiple prompts in compare_prompts.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

