"""
Advanced Prompt Usage - Fetch, Format, and Use with Gemini
===========================================================

Demonstrates advanced prompt operations:
1. Fetching prompts from MLflow UI
2. Extracting template content
3. Formatting with variables
4. Using with Gemini model for generation
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import google.generativeai as genai
from config import Config


def fetch_and_use_prompt_with_gemini():
    """
    Complete workflow: Fetch prompt from MLflow UI and use with Gemini
    """
    
    print("\n" + "="*80)
    print("ADVANCED PROMPT USAGE WITH GEMINI")
    print("="*80 + "\n")
    
    # Setup
    Config.setup_mlflow()
    
    if not Config.GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not found in .env file")
        return
    
    genai.configure(api_key=Config.GEMINI_API_KEY)
    
    # Step 1: Fetch prompt from MLflow
    print("📥 STEP 1: Fetching Prompt from MLflow Registry")
    print("-" * 80)
    
    prompt_name = os.getenv("MLFLOW_PROMPT_NAME", "email_generation_prompt")
    prompt_alias = os.getenv("MLFLOW_PROMPT_ALIAS", "champion")
    
    try:
        prompt_uri = f"prompts:/{prompt_name}@{prompt_alias}"
        print(f"Loading: {prompt_uri}")
        
        mlflow_prompt = mlflow.genai.load_prompt(prompt_uri)
        print(f"✅ Loaded: {mlflow_prompt.name} (v{mlflow_prompt.version})")
        
        # Extract template using multiple methods
        template_text = extract_template_robust(mlflow_prompt)
        print(f"✅ Extracted template ({len(template_text)} chars)")
        
    except Exception as e:
        print(f"⚠️ Could not load from registry: {e}")
        print("Using fallback template...")
        template_text = """
Write a {style} {content_type} about {topic}.

Context: {context}

Requirements:
- Tone: {tone}
- Length: {length}
- Audience: {audience}

Output:
"""
    
    # Step 2: Prepare variables
    print("\n📝 STEP 2: Preparing Template Variables")
    print("-" * 80)
    
    variables = {
        "style": "professional",
        "content_type": "email",
        "topic": "New MLflow Prompt Fetching Feature",
        "context": "We've implemented a new feature that allows dynamic prompt fetching from MLflow UI with multiple extraction methods",
        "tone": "enthusiastic and informative",
        "length": "2-3 paragraphs",
        "audience": "development team"
    }
    
    for key, value in variables.items():
        print(f"  {key}: {value}")
    
    # Step 3: Format template
    print("\n🔧 STEP 3: Formatting Template")
    print("-" * 80)
    
    try:
        formatted_prompt = template_text.format(**variables)
        print("✅ Template formatted successfully")
        print("\n📄 Formatted Prompt:")
        print("-" * 40)
        print(formatted_prompt)
        print("-" * 40)
    except Exception as e:
        print(f"⚠️ Formatting failed: {e}")
        formatted_prompt = template_text
    
    # Step 4: Generate with Gemini
    print("\n🤖 STEP 4: Generating Response with Gemini")
    print("-" * 80)
    
    with mlflow.start_run(run_name="prompt_fetch_and_generate"):
        # Log prompt info
        mlflow.log_param("prompt_name", prompt_name)
        mlflow.log_param("prompt_alias", prompt_alias)
        mlflow.log_param("model", Config.GEMINI_MODEL)
        
        # Generate response
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        
        print(f"Using model: {Config.GEMINI_MODEL}")
        print("Generating response...")
        
        try:
            response = model.generate_content(formatted_prompt)
            output = response.text
            
            print(f"\n✅ Generation successful!")
            print(f"📏 Output length: {len(output)} characters")
            
            # Log results
            mlflow.log_text(formatted_prompt, "input_prompt.txt")
            mlflow.log_text(output, "generated_output.txt")
            mlflow.log_metric("output_length", len(output))
            
            # Display output
            print("\n📤 Generated Output:")
            print("="*80)
            print(output)
            print("="*80)
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            mlflow.log_param("error", str(e))
    
    print("\n✅ Workflow complete!")


def extract_template_robust(prompt_obj):
    """
    Robust template extraction with multiple fallback methods
    
    Args:
        prompt_obj: MLflow prompt object
        
    Returns:
        str: Extracted template text
    """
    methods = [
        # Method 1: Direct string template
        lambda p: p.template if isinstance(p.template, str) else None,
        
        # Method 2: Dict with common keys
        lambda p: (p.template.get('text') or p.template.get('content') or 
                   p.template.get('template')) if isinstance(p.template, dict) else None,
        
        # Method 3: Object attributes
        lambda p: (getattr(p.template, 'text', None) or 
                   getattr(p.template, 'content', None)),
        
        # Method 4: Private attributes
        lambda p: getattr(p, '_template', None),
        
        # Method 5: Methods
        lambda p: p.to_string() if hasattr(p, 'to_string') else None,
        lambda p: p.get_template() if hasattr(p, 'get_template') else None,
        
        # Method 6: String representation
        lambda p: str(p.template)
    ]
    
    for i, method in enumerate(methods, 1):
        try:
            result = method(prompt_obj)
            if result:
                print(f"  ✅ Extracted using method {i}")
                return result
        except:
            continue
    
    raise ValueError("Could not extract template from prompt object")


def compare_prompt_versions():
    """
    Fetch and compare multiple versions of the same prompt
    """
    print("\n" + "="*80)
    print("COMPARING PROMPT VERSIONS")
    print("="*80 + "\n")
    
    Config.setup_mlflow()
    
    prompt_name = "email_generation_prompt"
    
    # Fetch different versions/aliases
    aliases = ["champion", "baseline", "challenger"]
    
    for alias in aliases:
        try:
            prompt_uri = f"prompts:/{prompt_name}@{alias}"
            print(f"\n📥 Fetching: {alias}")
            print(f"   URI: {prompt_uri}")
            
            prompt = mlflow.genai.load_prompt(prompt_uri)
            template = extract_template_robust(prompt)
            
            print(f"   ✅ Version: {prompt.version}")
            print(f"   📏 Length: {len(template)} chars")
            print(f"   📝 Preview: {template[:100]}...")
            
        except Exception as e:
            print(f"   ⚠️ Not available: {e}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Prompt Usage")
    parser.add_argument(
        "--mode",
        choices=["generate", "compare"],
        default="generate",
        help="Mode: 'generate' uses prompt with Gemini, 'compare' shows versions"
    )
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        fetch_and_use_prompt_with_gemini()
    else:
        compare_prompt_versions()

