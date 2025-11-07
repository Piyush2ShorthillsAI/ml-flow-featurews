#!/usr/bin/env python3
"""
MLflow Features Quick Start
Run this script to see a complete MLflow workflow in action
"""
import sys
import mlflow
from config import Config
from openai import OpenAI
from mlflow.genai import scorer
from mlflow.entities import Feedback


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"{text}")
    print(f"{'='*70}\n")


def main():
    print_header("🚀 MLflow Features Quick Start")
    
    # Validate configuration
    print("Checking configuration...")
    if not Config.OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY not set!")
        print("\nPlease:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key")
        print("3. Run this script again\n")
        return
    
    print("✅ Configuration valid\n")
    
    # Setup MLflow
    print("Setting up MLflow...")
    Config.setup_mlflow()
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    mlflow.openai.autolog()
    print("✅ MLflow ready\n")
    
    # Step 1: Register a Prompt
    print_header("Step 1: Register a Prompt")
    
    try:
        prompt = mlflow.genai.register_prompt(
            name="quickstart_prompt",
            template="You are a helpful assistant. Answer briefly: {{question}}",
            commit_message="Quickstart demo prompt"
        )
        print(f"✅ Registered prompt: quickstart_prompt (version {prompt.version})")
    except Exception as e:
        print(f"⚠️  Prompt may already exist: {e}")
        prompt = mlflow.genai.load_prompt("prompts:/quickstart_prompt@latest")
        print(f"✅ Loaded existing prompt (version {prompt.version})")
    
    # Step 2: Create Evaluation Function
    print_header("Step 2: Create Prediction Function")
    
    @mlflow.trace
    def predict_fn(question: str) -> str:
        """Generate answer using registered prompt"""
        prompt_obj = mlflow.genai.load_prompt("prompts:/quickstart_prompt@latest")
        formatted = prompt_obj.format(question=question)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": formatted}],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    print("✅ Prediction function created")
    
    # Test prediction
    print("\nTesting prediction...")
    test_result = predict_fn("What is MLflow?")
    print(f"Question: What is MLflow?")
    print(f"Answer: {test_result[:100]}...\n")
    
    # Step 3: Create Evaluation Dataset
    print_header("Step 3: Create Evaluation Dataset")
    
    eval_data = [
        {
            "inputs": {"question": "What is machine learning?"},
            "expectations": {"answer": "ML is a type of AI"}
        },
        {
            "inputs": {"question": "What is Python?"},
            "expectations": {"answer": "Python is a programming language"}
        },
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is an ML platform"}
        }
    ]
    
    print(f"✅ Created dataset with {len(eval_data)} samples")
    
    # Step 4: Define Scorers
    print_header("Step 4: Define Evaluation Scorers")
    
    @scorer
    def is_concise(outputs: str) -> Feedback:
        """Check if response is concise"""
        word_count = len(outputs.split())
        score = 1.0 if word_count <= 50 else 0.5
        return Feedback(
            value=score,
            rationale=f"Response is {word_count} words"
        )
    
    @scorer
    def is_not_empty(outputs: str) -> bool:
        """Check if response is not empty"""
        return bool(outputs and outputs.strip())
    
    scorers = [is_concise, is_not_empty]
    print(f"✅ Created {len(scorers)} scorers")
    
    # Step 5: Run Evaluation
    print_header("Step 5: Run Evaluation")
    
    print("Running evaluation (this may take a moment)...")
    
    with mlflow.start_run(run_name="quickstart_demo"):
        # Log parameters
        mlflow.log_param("prompt_name", "quickstart_prompt")
        mlflow.log_param("model", "gpt-4o-mini")
        mlflow.log_param("num_samples", len(eval_data))
        mlflow.set_tag("demo_type", "quickstart")
        
        # Run evaluation
        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=scorers
        )
        
        print(f"\n✅ Evaluation complete!")
        
        # Display results
        if hasattr(results, 'metrics'):
            print(f"\n📊 Metrics:")
            for metric_name, value in results.metrics.items():
                if isinstance(value, float):
                    print(f"   {metric_name}: {value:.4f}")
                else:
                    print(f"   {metric_name}: {value}")
    
    # Final Summary
    print_header("✨ Quick Start Complete!")
    
    print("You've successfully:")
    print("  ✅ Registered a prompt")
    print("  ✅ Created a prediction function with tracing")
    print("  ✅ Built an evaluation dataset")
    print("  ✅ Defined custom scorers")
    print("  ✅ Run a complete evaluation")
    
    print(f"\n🌐 View results in MLflow UI:")
    print(f"   {Config.MLFLOW_TRACKING_URI}")
    print(f"\n📚 Next Steps:")
    print(f"   1. Explore the examples in each folder")
    print(f"   2. Check GETTING_STARTED.md for guided learning")
    print(f"   3. Read README.md for detailed documentation")
    
    print_header("Happy MLflow-ing! 🎉")


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
        print("  2. Check your .env file has OPENAI_API_KEY set")
        print("  3. Ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)

