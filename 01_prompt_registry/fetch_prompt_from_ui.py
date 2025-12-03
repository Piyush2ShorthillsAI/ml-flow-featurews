"""
Fetch Prompt from MLflow UI - Advanced Template Extraction
===========================================================

This script demonstrates how to fetch prompts registered in MLflow UI
and extract their template content using multiple fallback methods.

Features:
- Load prompts by alias or version
- Comprehensive template extraction with multiple methods
- Full object inspection and debugging
- Support for various prompt formats
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from config import Config


class PromptFetcher:
    """Advanced prompt fetcher with multiple extraction methods"""
    
    def __init__(self):
        """Initialize the prompt fetcher"""
        Config.setup_mlflow()
    
    def inspect_prompt_object(self, prompt_obj):
        """
        Comprehensively inspect a prompt object to understand its structure
        
        Args:
            prompt_obj: MLflow prompt object to inspect
        """
        print("\n" + "="*80)
        print("🔍 FULL PROMPT OBJECT INSPECTION")
        print("="*80)
        print(f"Object type: {type(prompt_obj)}")
        print(f"Object class: {prompt_obj.__class__.__name__}")
        
        print(f"\n📋 Basic Info:")
        print(f"  Name: {getattr(prompt_obj, 'name', 'N/A')}")
        print(f"  Version: {getattr(prompt_obj, 'version', 'N/A')}")
        
        print(f"\n📝 All Non-Private Attributes:")
        for attr in dir(prompt_obj):
            if not attr.startswith('_'):
                try:
                    val = getattr(prompt_obj, attr)
                    if not callable(val):
                        val_str = str(val)[:100] if val else "None"
                        print(f"  • {attr}: {type(val).__name__} = {val_str}")
                except Exception as e:
                    print(f"  • {attr}: Error - {e}")
        
        print("="*80 + "\n")
    
    def extract_template(self, prompt_obj):
        """
        Extract template text from prompt object using multiple fallback methods
        
        Args:
            prompt_obj: MLflow prompt object
            
        Returns:
            str: Extracted template text
        """
        print("🔄 Attempting to extract template using multiple methods...\n")
        
        prompt_text = None
        extraction_method = None
        
        # Method 1: Direct template attribute (string)
        try:
            if hasattr(prompt_obj, 'template'):
                template = prompt_obj.template
                if isinstance(template, str):
                    prompt_text = template
                    extraction_method = "Method 1: Direct .template string attribute"
                    print(f"✅ {extraction_method}")
        except Exception as e:
            print(f"❌ Method 1 failed: {e}")
        
        # Method 2: Template as dict with content key
        if prompt_text is None:
            try:
                template_obj = prompt_obj.template
                if isinstance(template_obj, dict):
                    # Try common dict keys
                    for key in ['text', 'content', 'template', 'prompt', 'value']:
                        if key in template_obj:
                            prompt_text = template_obj[key]
                            extraction_method = f"Method 2: template dict with '{key}' key"
                            print(f"✅ {extraction_method}")
                            break
            except Exception as e:
                print(f"❌ Method 2 failed: {e}")
        
        # Method 3: Template object with text/content attributes
        if prompt_text is None:
            try:
                template_obj = prompt_obj.template
                for attr in ['text', 'content', 'template', 'prompt']:
                    if hasattr(template_obj, attr):
                        prompt_text = getattr(template_obj, attr)
                        extraction_method = f"Method 3: template.{attr} attribute"
                        print(f"✅ {extraction_method}")
                        break
            except Exception as e:
                print(f"❌ Method 3 failed: {e}")
        
        # Method 4: Private _template attribute
        if prompt_text is None:
            try:
                if hasattr(prompt_obj, '_template'):
                    prompt_text = prompt_obj._template
                    extraction_method = "Method 4: Private _template attribute"
                    print(f"✅ {extraction_method}")
            except Exception as e:
                print(f"❌ Method 4 failed: {e}")
        
        # Method 5: to_string() or similar methods
        if prompt_text is None:
            try:
                for method_name in ['to_string', 'get_template', 'render', '__str__']:
                    if hasattr(prompt_obj, method_name):
                        method = getattr(prompt_obj, method_name)
                        if callable(method):
                            prompt_text = method()
                            extraction_method = f"Method 5: {method_name}() method"
                            print(f"✅ {extraction_method}")
                            break
            except Exception as e:
                print(f"❌ Method 5 failed: {e}")
        
        # Method 6: String representation fallback
        if prompt_text is None:
            try:
                prompt_text = str(prompt_obj.template)
                extraction_method = "Method 6: Fallback str(template)"
                print(f"⚠️ {extraction_method}")
            except Exception as e:
                print(f"❌ Method 6 failed: {e}")
                raise ValueError("Could not extract template from prompt object")
        
        return prompt_text, extraction_method
    
    def fetch_prompt_by_alias(self, prompt_name, alias="champion"):
        """
        Fetch a prompt from MLflow registry using alias
        
        Args:
            prompt_name (str): Name of the prompt in registry
            alias (str): Alias to use (e.g., "champion", "baseline")
            
        Returns:
            tuple: (prompt_text, prompt_metadata)
        """
        print("\n" + "="*80)
        print(f"📥 FETCHING PROMPT FROM MLFLOW REGISTRY")
        print("="*80)
        
        # Construct URI
        
        prompt_uri = f"prompts:/comprehensive_technical_writer@local"
        print(f"📍 URI: {prompt_uri}")
        print(f"📝 Prompt Name: {prompt_name}")
        print(f"🏷️  Alias: {alias}")
        
        try:
            # Load prompt from MLflow
            mlflow_prompt = mlflow.genai.load_prompt(prompt_uri)
            print(f"✅ Successfully loaded prompt")
            
            # Inspect the prompt object
            self.inspect_prompt_object(mlflow_prompt)
            
            # Extract template
            template_text, method = self.extract_template(mlflow_prompt)
            
            # Display results
            print("\n" + "="*80)
            print("📋 EXTRACTED TEMPLATE CONTENT")
            print("="*80)
            print(f"Extraction Method: {method}")
            print(f"Template Length: {len(template_text)} characters")
            print(f"Template Type: {type(template_text).__name__}")
            print("\n--- TEMPLATE START ---")
            print(template_text)
            print("--- TEMPLATE END ---\n")
            print("="*80)
            
            # Prepare metadata
            metadata = {
                "name": getattr(mlflow_prompt, 'name', None),
                "version": getattr(mlflow_prompt, 'version', None),
                "alias": alias,
                "uri": prompt_uri,
                "extraction_method": method,
                "template_length": len(template_text)
            }
            
            return template_text, metadata
            
        except Exception as e:
            print(f"\n❌ Error fetching prompt: {str(e)}")
            raise
    
    def fetch_prompt_by_version(self, prompt_name, version):
        """
        Fetch a specific version of a prompt
        
        Args:
            prompt_name (str): Name of the prompt in registry
            version (int): Version number
            
        Returns:
            tuple: (prompt_text, prompt_metadata)
        """
        prompt_uri = f"prompts:/{prompt_name}/{version}"
        print(f"\n📥 Fetching prompt version {version}: {prompt_uri}")
        
        mlflow_prompt = mlflow.genai.load_prompt(prompt_uri)
        template_text, method = self.extract_template(mlflow_prompt)
        
        metadata = {
            "name": prompt_name,
            "version": version,
            "uri": prompt_uri,
            "extraction_method": method
        }
        
        return template_text, metadata
    
    def format_template_with_variables(self, template_text, variables):
        """
        Format a template with provided variables
        
        Args:
            template_text (str): Template with placeholders
            variables (dict): Variables to substitute
            
        Returns:
            str: Formatted template
        """
        print("\n🔧 Formatting template with variables...")
        print(f"Variables: {variables}")
        
        try:
            formatted = template_text.format(**variables)
            print("✅ Template formatted successfully")
            return formatted
        except KeyError as e:
            print(f"⚠️ Missing variable: {e}")
            raise
        except Exception as e:
            print(f"❌ Formatting error: {e}")
            raise


def main():
    """Main demonstration function"""
    
    print("\n" + "="*80)
    print("MLFLOW PROMPT FETCHING - COMPREHENSIVE EXAMPLE")
    print("="*80 + "\n")
    
    # Initialize fetcher
    fetcher = PromptFetcher()
    
    # Example 1: Fetch prompt by alias
    print("\n" + "🎯 EXAMPLE 1: Fetch Prompt by Alias")
    print("-" * 80)
    
    try:
        # Get prompt name from environment or use default
       
        template_text, metadata = fetcher.fetch_prompt_by_alias(
            prompt_name="comprehensive_technical_writer",
            alias="local"
        )
        
        print(f"\n✅ Successfully fetched prompt:")
        print(f"   Name: {metadata['name']}")
        print(f"   Version: {metadata['version']}")
        print(f"   Length: {metadata['template_length']} chars")
        
        # Example 2: Format the template with variables
        print("\n" + "🎯 EXAMPLE 2: Format Template with Variables")
        print("-" * 80)
        
        # Check if template has placeholders
        if '{' in template_text and '}' in template_text:
            print("Template contains placeholders, formatting...")
            
            # Example variables (adjust based on your template)
            example_vars = {
                "topic": "MLflow Integration",
                "tone": "professional",
                "recipient": "Team",
                "context": "Announcing new prompt fetching feature"
            }
            
            try:
                formatted = fetcher.format_template_with_variables(
                    template_text, 
                    example_vars
                )
                print("\n📄 Formatted Output:")
                print("-" * 80)
                print(formatted)
                print("-" * 80)
            except KeyError as e:
                print(f"⚠️ Template requires variable: {e}")
                print("Please check the template placeholders and provide correct variables")
        else:
            print("Template has no placeholders, using as-is")
        
    except mlflow.exceptions.MlflowException as e:
        print(f"\n⚠️ MLflow Error: {str(e)}")
        print("\n💡 Possible reasons:")
        print("   1. Prompt not registered in MLflow")
        print("   2. Alias doesn't exist")
        print("   3. MLflow tracking URI not configured")
        print("\n📝 To register a prompt, run: python 01_prompt_registry/register_prompts.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ PROMPT FETCHING DEMONSTRATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

