"""
Example: Complete Prompt Lifecycle Management
Demonstrates creating, versioning, aliasing, and managing prompt lifecycle
"""
import sys
sys.path.append('..')

import mlflow
from config import Config

def main():
    """Demonstrate complete prompt lifecycle"""
    
    print("\n" + "="*70)
    print("MLflow Prompt Lifecycle Management")
    print("="*70 + "\n")
    
    # Setup MLflow
    Config.setup_mlflow()
    
    prompt_name = "customer_support_prompt"
    
    # ============================================================
    # 1. Create Initial Prompt Version
    # ============================================================
    print("1️⃣  Creating initial prompt version...")
    
    v1_template = [
        {
            "role": "system",
            "content": "You are a customer support agent. Answer customer questions."
        },
        {
            "role": "user",
            "content": "{{customer_question}}"
        }
    ]
    
    prompt_v1 = mlflow.genai.register_prompt(
        name=prompt_name,
        template=v1_template,
        commit_message="Initial customer support prompt",
        tags={"stage": "development", "quality": "draft"}
    )
    
    print(f"✅ Created version 1")
    print(f"   Prompt: {prompt_name}")
    print(f"   Version: {prompt_v1.version}\n")
    
    # ============================================================
    # 2. Improve and Create Version 2
    # ============================================================
    print("2️⃣  Creating improved version...")
    
    v2_template = [
        {
            "role": "system",
            "content": "You are a professional customer support agent. "
                      "Answer customer questions politely and provide helpful solutions. "
                      "If you don't know the answer, guide them to the right resource."
        },
        {
            "role": "user",
            "content": "Customer question: {{customer_question}}"
        }
    ]
    
    prompt_v2 = mlflow.genai.register_prompt(
        name=prompt_name,
        template=v2_template,
        commit_message="Improved with better instructions and tone",
        tags={"stage": "testing", "quality": "improved"}
    )
    
    print(f"✅ Created version 2")
    print(f"   Version: {prompt_v2.version}\n")
    
    # ============================================================
    # 3. Promote to Staging
    # ============================================================
    print("3️⃣  Promoting version 2 to staging...")
    
    mlflow.genai.set_prompt_alias(
        name=prompt_name,
        alias="staging",
        version=prompt_v2.version
    )
    
    print(f"✅ Set 'staging' alias to version {prompt_v2.version}")
    print(f"   Now testing in staging environment...\n")
    
    # ============================================================
    # 4. After Testing - Promote to Production
    # ============================================================
    print("4️⃣  Promoting to production after successful testing...")
    
    # Move production alias from v1 to v2
    mlflow.genai.set_prompt_alias(
        name=prompt_name,
        alias="production",
        version=prompt_v2.version
    )
    
    mlflow.genai.set_prompt_alias(
        name=prompt_name,
        alias="latest",
        version=prompt_v2.version
    )
    
    print(f"✅ Set 'production' alias to version {prompt_v2.version}")
    print(f"✅ Set 'latest' alias to version {prompt_v2.version}\n")
    
    # ============================================================
    # 5. Create Version 3 with Additional Features
    # ============================================================
    print("5️⃣  Creating version 3 with context awareness...")
    
    v3_template = [
        {
            "role": "system",
            "content": "You are a professional customer support agent for {{company_name}}. "
                      "Answer questions politely using the knowledge base: {{knowledge_base}}. "
                      "Provide helpful solutions and escalate if needed."
        },
        {
            "role": "user",
            "content": "Customer question: {{customer_question}}"
        }
    ]
    
    prompt_v3 = mlflow.genai.register_prompt(
        name=prompt_name,
        template=v3_template,
        commit_message="Added context variables for company and knowledge base",
        tags={"stage": "development", "quality": "advanced"}
    )
    
    print(f"✅ Created version 3")
    print(f"   Version: {prompt_v3.version}")
    print(f"   New features: company context, knowledge base integration\n")
    
    # ============================================================
    # 6. Test New Version in Staging
    # ============================================================
    print("6️⃣  Moving version 3 to staging for testing...")
    
    mlflow.genai.set_prompt_alias(
        name=prompt_name,
        alias="staging",
        version=prompt_v3.version
    )
    
    print(f"✅ Updated 'staging' to version {prompt_v3.version}")
    print(f"   Production remains on version {prompt_v2.version}\n")
    
    # ============================================================
    # 7. Demonstrate Rollback Scenario
    # ============================================================
    print("7️⃣  Rollback scenario (if version 3 has issues)...")
    
    # Rollback staging to version 2
    mlflow.genai.set_prompt_alias(
        name=prompt_name,
        alias="staging",
        version=prompt_v2.version
    )
    
    print(f"✅ Rolled back 'staging' to version {prompt_v2.version}")
    print(f"   Can iterate on version 3 without affecting staging/production\n")
    
    # ============================================================
    # 8. Load and Compare Versions
    # ============================================================
    print("8️⃣  Loading and comparing different versions...")
    
    try:
        prod_prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@production")
        staging_prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@staging")
        latest_prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@latest")
        
        print(f"✅ Production version: {prod_prompt.version}")
        print(f"✅ Staging version: {staging_prompt.version}")
        print(f"✅ Latest version: {latest_prompt.version}\n")
    except Exception as e:
        print(f"⚠️  Error loading prompts: {e}\n")
    
    # ============================================================
    # Summary
    # ============================================================
    print("="*70)
    print("✨ Prompt Lifecycle Summary")
    print("="*70)
    print(f"Prompt: {prompt_name}")
    print(f"Total versions created: 3")
    print(f"\nVersion history:")
    print(f"  v1: Initial draft")
    print(f"  v2: Improved instructions → PRODUCTION")
    print(f"  v3: Added context features → DEVELOPMENT")
    print(f"\nAliases:")
    print(f"  production: v{prompt_v2.version}")
    print(f"  staging: v{prompt_v2.version} (after rollback)")
    print(f"  latest: v{prompt_v2.version}")
    print(f"\n💡 This demonstrates safe iteration while keeping production stable")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

