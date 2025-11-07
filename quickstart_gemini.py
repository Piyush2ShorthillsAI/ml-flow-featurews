#!/usr/bin/env python3
"""
MLflow Features Quick Start with Gemini 2.5 Pro
Run this script to see a complete MLflow workflow using Google Gemini
"""
import sys
import mlflow
from config import Config
import google.genai as genai
from mlflow.genai import scorer
from mlflow.entities import Feedback


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"{text}")
    print(f"{'='*70}\n")


def main():
    print_header("🚀 MLflow Features Quick Start with Gemini 2.5 Pro")
    
    # Validate configuration
    print("Checking configuration...")
    if not Config.GEMINI_API_KEY:
        print("❌ ERROR: GEMINI_API_KEY not set!")
        print("\nPlease:")
        print("1. Copy .env.example to .env")
        print("2. Add your Gemini API key: GEMINI_API_KEY=your-key-here")
        print("3. Run this script again\n")
        return
    
    print("✅ Configuration valid")
    print(f"   Using model: {Config.GEMINI_MODEL}\n")
    
    # Setup MLflow
    print("Setting up MLflow...")
    Config.setup_mlflow()
    client = genai.Client(api_key=Config.GEMINI_API_KEY)
    mlflow.gemini.autolog()
    print("✅ MLflow ready with Gemini auto-tracing\n")
    
    # Step 1: Register a Prompt
    print_header("Step 1: Register a Prompt")
    
    try:
        prompt = mlflow.genai.register_prompt(
            name="gemini_quickstart_prompt",
            template="You are a helpful AI assistant powered by Gemini 2.5 Pro. Answer concisely and accurately: {{question}}",
            commit_message="Quickstart demo prompt for Gemini",
            tags={"model": "gemini-2.5-pro", "version": "v1"}
        )
        print(f"✅ Registered prompt: gemini_quickstart_prompt (version {prompt.version})")
    except Exception as e:
        print(f"⚠️  Prompt may already exist: {e}")
        prompt = mlflow.genai.load_prompt("prompts:/gemini_quickstart_prompt@latest")
        print(f"✅ Loaded existing prompt (version {prompt.version})")
    
    # Step 2: Create Evaluation Function
    print_header("Step 2: Create Prediction Function with Gemini")
    
    @mlflow.trace
    def predict_fn(question: str) -> str:
        """Generate answer using Gemini 2.5 Pro"""
        prompt_obj = mlflow.genai.load_prompt("prompts:/gemini_quickstart_prompt@latest")
        formatted = prompt_obj.format(question=question)
        
        response = client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=formatted
        )
        return response.text
    
    print("✅ Prediction function created with Gemini 2.5 Pro")
    
    # Test prediction
    print("\nTesting prediction...")
    test_result = predict_fn("What is MLflow and why is it important?")
    print(f"Question: What is MLflow and why is it important?")
    print(f"Answer: {test_result[:150]}...\n")
    
    # Step 3: Create Evaluation Dataset
    print_header("Step 3: Create Evaluation Dataset")
    
    eval_data = [
        {
            "inputs": {"question": "What is machine learning?"},
            "expectations": {"answer": "ML is a type of AI that learns from data"}
        },
        {
            "inputs": {"question": "What is Python used for?"},
            "expectations": {"answer": "Python is used for data science, web development, and AI"}
        },
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is a platform for ML lifecycle management"}
        },
        {
            "inputs": {"question": "What are the benefits of prompt engineering?"},
            "expectations": {"answer": "Better AI responses, cost efficiency, and improved quality"}
        }
    ]
    
    print(f"✅ Created dataset with {len(eval_data)} samples")
    
    # Step 4: Define Scorers
    print_header("Step 4: Define Evaluation Scorers")
    
    @scorer
    def is_concise(outputs: str) -> Feedback:
        """Check if response is concise"""
        word_count = len(outputs.split())
        score = 1.0 if word_count <= 80 else 0.5
        return Feedback(
            value=score,
            rationale=f"Response is {word_count} words (target: ≤80)"
        )
    
    @scorer
    def is_not_empty(outputs: str) -> bool:
        """Check if response is not empty"""
        return bool(outputs and outputs.strip())
    
    @scorer
    def has_relevant_content(outputs: str, inputs: dict) -> Feedback:
        """Check if response is relevant to the question"""
        question = inputs.get('question', '').lower()
        output_lower = outputs.lower()
        
        # Extract key terms from question
        key_terms = [word for word in question.split() if len(word) > 4]
        found_terms = [term for term in key_terms if term in output_lower]
        
        score = min(len(found_terms) / max(len(key_terms), 1), 1.0)
        
        return Feedback(
            value=score,
            rationale=f"Found {len(found_terms)}/{len(key_terms)} key terms from question"
        )
    
    scorers = [is_concise, is_not_empty, has_relevant_content]
    print(f"✅ Created {len(scorers)} scorers")
    
    # Step 5: Run Evaluation
    print_header("Step 5: Run Evaluation with Gemini 2.5 Pro")
    
    print("Running evaluation (this may take a moment)...")
    print("Gemini 2.5 Pro processing your queries...\n")
    
    with mlflow.start_run(run_name="gemini_quickstart_demo"):
        # Log parameters
        mlflow.log_param("prompt_name", "gemini_quickstart_prompt")
        mlflow.log_param("model", Config.GEMINI_MODEL)
        mlflow.log_param("num_samples", len(eval_data))
        mlflow.log_param("llm_provider", "gemini")
        mlflow.set_tag("demo_type", "quickstart_gemini")
        
        # Run evaluation
        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=scorers
        )
        
        print(f"✅ Evaluation complete!")
        
        # Display results
        if hasattr(results, 'metrics'):
            print(f"\n📊 Metrics:")
            for metric_name, value in results.metrics.items():
                if isinstance(value, float):
                    print(f"   {metric_name}: {value:.4f}")
                else:
                    print(f"   {metric_name}: {value}")
        
        # Token usage summary
        trace_id = mlflow.get_last_active_trace_id()
        if trace_id:
            trace = mlflow.get_trace(trace_id)
            if hasattr(trace.info, 'token_usage') and trace.info.token_usage:
                usage = trace.info.token_usage
                print(f"\n🪙 Token Usage (last call):")
                print(f"   Input: {usage.get('input_tokens', 0)}")
                print(f"   Output: {usage.get('output_tokens', 0)}")
                print(f"   Total: {usage.get('total_tokens', 0)}")
    
    # Final Summary
    print_header("✨ Quick Start Complete with Gemini 2.5 Pro!")
    
    print("You've successfully:")
    print("  ✅ Registered a prompt for Gemini 2.5 Pro")
    print("  ✅ Created a prediction function with Gemini")
    print("  ✅ Built an evaluation dataset")
    print("  ✅ Defined custom scorers")
    print("  ✅ Run a complete evaluation")
    print("  ✅ Tracked token usage")
    
    print(f"\n🌐 View results in MLflow UI:")
    print(f"   {Config.MLFLOW_TRACKING_URI}")
    
    print(f"\n🤖 Gemini 2.5 Pro Benefits:")
    print(f"   • Large context window (up to 2M tokens)")
    print(f"   • Advanced reasoning and analysis")
    print(f"   • Multimodal capabilities")
    print(f"   • Cost-effective for complex tasks")
    
    print(f"\n📚 Next Steps:")
    print(f"   1. Run: python 05_tracing/gemini_tracing.py")
    print(f"   2. Check GETTING_STARTED.md for more examples")
    print(f"   3. Explore Gemini-specific features")
    
    print_header("Happy MLflow-ing with Gemini! 🎉")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nQuickstart interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure MLflow server is running: mlflow ui --port 5000")
        print("  2. Check your .env file has GEMINI_API_KEY set")
        print("  3. Ensure google-genai is installed: pip install google-genai")
        print("  4. Verify your Gemini API key is valid")
        sys.exit(1)

