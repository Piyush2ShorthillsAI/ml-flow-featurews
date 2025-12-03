"""
Simple Prompt Fetching by Alias - Direct Example
=================================================

This script demonstrates the exact pattern for fetching prompts
from MLflow UI using alias names.

Usage:
    export MLFLOW_PROMPT_ALIAS="champion"
    python simple_fetch_by_alias.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from config import Config


def main():
    """Simple example of loading prompt by alias"""
    
    print("\n" + "="*80)
    print("SIMPLE PROMPT LOADING BY ALIAS")
    print("="*80 + "\n")
    
    # Setup MLflow
    Config.setup_mlflow()
    
    # Get alias from environment (or use default)
    prompt_alias = os.getenv("MLFLOW_PROMPT_ALIAS", "champion")
    prompt_name = os.getenv("MLFLOW_PROMPT_NAME", "qa_prompt_chat")
    
    # Construct URI with alias
    prompt_uri = f"prompts:/{prompt_name}@{prompt_alias}"
    
    print(f"📝 Loading prompt from: {prompt_uri}")
    print(f"   Prompt name: {prompt_name}")
    print(f"   Alias: {prompt_alias}\n")
    
    try:
        # Load the prompt
        mlflow_prompt = mlflow.genai.load_prompt(prompt_uri)
        
        # Display basic info
        print(f"✅ Loaded MLflow prompt successfully!")
        print(f"   Type: {type(mlflow_prompt)}")
        print(f"   Name: {mlflow_prompt.name}")
        print(f"   Version: {mlflow_prompt.version}")
        
        # Access the template
        print(f"\n📋 Template Info:")
        print(f"   Template type: {type(mlflow_prompt.template)}")
        
        if isinstance(mlflow_prompt.template, str):
            # Simple string template
            print(f"   Template length: {len(mlflow_prompt.template)} characters")
            print(f"\n📄 Template Content:")
            print("-" * 80)
            print(mlflow_prompt.template)
            print("-" * 80)
            
        elif isinstance(mlflow_prompt.template, list):
            # Chat-style template
            print(f"   Number of messages: {len(mlflow_prompt.template)}")
            print(f"\n📄 Template Content (Chat Format):")
            print("-" * 80)
            for i, msg in enumerate(mlflow_prompt.template, 1):
                print(f"\nMessage {i}:")
                print(f"  Role: {msg.get('role', 'N/A')}")
                print(f"  Content: {msg.get('content', 'N/A')[:200]}")
            print("-" * 80)
            
        elif isinstance(mlflow_prompt.template, dict):
            # Dictionary template
            print(f"   Keys: {list(mlflow_prompt.template.keys())}")
            print(f"\n📄 Template Content:")
            print("-" * 80)
            for key, value in mlflow_prompt.template.items():
                print(f"{key}: {value}")
            print("-" * 80)
        
        else:
            # Other format
            print(f"\n📄 Template Content:")
            print("-" * 80)
            print(str(mlflow_prompt.template))
            print("-" * 80)
        
        # Show how to use with variables
        print(f"\n💡 Usage Example:")
        print("-" * 80)
        print("# If template has variables like {topic}, {tone}, etc:")
        print("variables = {")
        print("    'topic': 'MLflow Updates',")
        print("    'tone': 'professional',")
        print("    'recipient': 'Team'")
        print("}")
        print("\n# Format the template:")
        print("formatted_prompt = template.format(**variables)")
        print("-" * 80)
        
        print(f"\n✅ Success! Prompt loaded from MLflow UI\n")
        
    except mlflow.exceptions.MlflowException as e:
        print(f"\n❌ MLflow Error: {str(e)}\n")
        
        if "not found" in str(e).lower():
            print("💡 Possible solutions:")
            print("   1. Check if prompt is registered in MLflow")
            print("   2. Verify the alias exists")
            print("   3. Try using version number instead of alias:")
            print(f"      prompts:/{prompt_name}/1")
            print("\n📝 Available prompts on your server:")
            print("   - qa_prompt_chat (versions 5, 6)")
            print("   - qa_prompt_simple (version 3)")
            print("   - summarization_prompt (version 3)")
            print("   - sentiment_classifier (version 3)")
            print("\n🔧 To use these prompts:")
            print("   export MLFLOW_PROMPT_NAME='qa_prompt_chat'")
            print("   python simple_fetch_by_alias.py")
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}\n")
        import traceback
        traceback.print_exc()


def fetch_by_version_example():
    """Example: Fetch by specific version instead of alias"""
    
    print("\n" + "="*80)
    print("ALTERNATIVE: FETCH BY VERSION NUMBER")
    print("="*80 + "\n")
    
    Config.setup_mlflow()
    
    prompt_name = "qa_prompt_chat"
    version = 6
    
    # Construct URI with version
    prompt_uri = f"prompts:/{prompt_name}/{version}"
    
    print(f"📝 Loading prompt: {prompt_uri}\n")
    
    try:
        mlflow_prompt = mlflow.genai.load_prompt(prompt_uri)
        
        print(f"✅ Loaded successfully!")
        print(f"   Name: {mlflow_prompt.name}")
        print(f"   Version: {mlflow_prompt.version}")
        print(f"   Template type: {type(mlflow_prompt.template)}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple prompt loading examples")
    parser.add_argument(
        "--mode",
        choices=["alias", "version"],
        default="alias",
        help="Loading mode: 'alias' or 'version'"
    )
    
    args = parser.parse_args()
    
    if args.mode == "alias":
        main()
    else:
        fetch_by_version_example()

