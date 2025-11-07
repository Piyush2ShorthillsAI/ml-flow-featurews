"""
Example: Load Prompts from MLflow Registry
Demonstrates how to load prompts by version, alias, and format them with variables
"""
import sys
sys.path.append('..')

import mlflow
from config import Config

def main():
    """Load prompts from MLflow Prompt Registry"""
    
    print("\n" + "="*70)
    print("MLflow Prompt Registry - Loading Example")
    print("="*70 + "\n")
    
    # Setup MLflow
    Config.setup_mlflow()
    
    # ============================================================
    # 1. Load Prompt by Specific Version
    # ============================================================
    print("1️⃣  Loading prompt by specific version...")
    
    try:
        prompt_v1 = mlflow.genai.load_prompt("prompts:/qa_prompt_chat/1")
        print(f"✅ Loaded: {prompt_v1.name} (Version {prompt_v1.version})")
        print(f"   Template type: {type(prompt_v1.template)}")
        print(f"   Number of messages: {len(prompt_v1.template) if isinstance(prompt_v1.template, list) else 'N/A'}\n")
    except Exception as e:
        print(f"⚠️  Could not load prompt version 1: {e}")
        print("   Run register_prompts.py first!\n")
    
    # ============================================================
    # 2. Load Prompt by Alias
    # ============================================================
    print("2️⃣  Loading prompts by alias...")
    
    try:
        # Load using 'latest' alias
        prompt_latest = mlflow.genai.load_prompt("prompts:/qa_prompt_chat@latest")
        print(f"✅ Loaded with 'latest' alias: Version {prompt_latest.version}")
        
        # Load using 'production' alias
        prompt_prod = mlflow.genai.load_prompt("prompts:/qa_prompt_chat@production")
        print(f"✅ Loaded with 'production' alias: Version {prompt_prod.version}")
        
        # Load using 'staging' alias
        prompt_staging = mlflow.genai.load_prompt("prompts:/qa_prompt_chat@staging")
        print(f"✅ Loaded with 'staging' alias: Version {prompt_staging.version}\n")
    except Exception as e:
        print(f"⚠️  Could not load prompts by alias: {e}\n")
    
    # ============================================================
    # 3. Format Prompt with Variables
    # ============================================================
    print("3️⃣  Formatting prompts with variables...")
    
    try:
        prompt = mlflow.genai.load_prompt("prompts:/qa_prompt_chat@latest")
        
        # Format with a question
        formatted = prompt.format(question="What is machine learning?")
        
        print(f"✅ Formatted prompt:")
        if isinstance(formatted, list):
            for i, msg in enumerate(formatted):
                print(f"\n   Message {i+1} ({msg['role']}):")
                print(f"   {msg['content'][:100]}...")
        else:
            print(f"   {formatted[:200]}...\n")
    except Exception as e:
        print(f"⚠️  Could not format prompt: {e}\n")
    
    # ============================================================
    # 4. Load and Format Different Prompt Types
    # ============================================================
    print("4️⃣  Loading specialized prompts...")
    
    prompt_names = [
        "qa_prompt_simple",
        "summarization_prompt",
        "sentiment_classifier",
        "cot_reasoning_prompt"
    ]
    
    for prompt_name in prompt_names:
        try:
            prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@latest")
            print(f"✅ Loaded: {prompt.name}")
            print(f"   Template preview: {str(prompt.template)[:80]}...")
            
            # Format with sample data
            if "summarization" in prompt_name:
                formatted = prompt.format(
                    text="This is a sample text to summarize.",
                    num_sentences=1
                )
            elif "sentiment" in prompt_name:
                formatted = prompt.format(text="I love this product!")
            elif "cot" in prompt_name:
                formatted = prompt.format(question="What is 2+2?")
            else:
                formatted = prompt.format(question="Test question")
            
            print(f"   Formatted length: {len(str(formatted))} chars\n")
        except Exception as e:
            print(f"⚠️  Could not load {prompt_name}: {e}\n")
    
    # ============================================================
    # 5. Convert to Single-Brace Format (for LangChain)
    # ============================================================
    print("5️⃣  Converting to single-brace format...")
    
    try:
        prompt = mlflow.genai.load_prompt("prompts:/qa_prompt_chat@latest")
        
        # MLflow uses {{variable}} format
        # Convert to {variable} format for LangChain
        single_brace = prompt.to_single_brace_format()
        
        print(f"✅ Converted to single-brace format")
        if isinstance(single_brace, list):
            print(f"   First message: {single_brace[0]['content'][:80]}...\n")
    except Exception as e:
        print(f"⚠️  Could not convert format: {e}\n")
    
    # ============================================================
    # Summary
    # ============================================================
    print("="*70)
    print("✨ Summary")
    print("="*70)
    print(f"✅ Loaded prompts by version number")
    print(f"✅ Loaded prompts by aliases (latest, production, staging)")
    print(f"✅ Formatted prompts with variables")
    print(f"✅ Converted to LangChain-compatible format")
    print(f"\n💡 Next: Try prompt evaluation in ../02_prompt_evaluation/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

