"""
Example: Model Versioning and Lifecycle Management
Demonstrates complete model lifecycle from development to production
"""
import sys
sys.path.append('..')

import mlflow
import mlflow.pyfunc
from config import Config
from mlflow import MlflowClient


def print_model_versions(client, model_name):
    """Helper to print all model versions"""
    versions = client.search_model_versions(f"name='{model_name}'")
    
    if not versions:
        print(f"   No versions found for {model_name}")
        return
    
    print(f"\n   Model Versions for '{model_name}':")
    print(f"   {'Version':<10} {'Stage':<15} {'Status':<15} {'Tags'}")
    print(f"   {'-'*60}")
    
    for v in sorted(versions, key=lambda x: int(x.version)):
        tags = ""
        if v.run_id:
            run = client.get_run(v.run_id)
            variant = run.data.tags.get('variant', 'base')
            temp = run.data.params.get('temperature', 'N/A')
            tags = f"variant={variant}, temp={temp}"
        
        print(f"   {v.version:<10} {v.current_stage:<15} {v.status:<15} {tags}")
    print()


def main():
    """Demonstrate model versioning and lifecycle"""
    
    print("\n" + "="*70)
    print("Model Versioning and Lifecycle Management")
    print("="*70 + "\n")
    
    # Validate config
    if not Config.GEMINI_API_KEY:
        print("⚠️  ERROR: GEMINI_API_KEY not set")
        return
    
    # Setup
    Config.setup_mlflow()
    client = MlflowClient()
    model_name = "gemini_qa_model"
    
    # ============================================================
    # 1. View Current Model Versions
    # ============================================================
    print("1️⃣  Current model versions:")
    
    try:
        print_model_versions(client, model_name)
    except Exception as e:
        print(f"⚠️  Error: {e}")
        print("   Run register_gemini_model.py first!\n")
        return
    
    # ============================================================
    # 2. Transition Models Through Stages
    # ============================================================
    print("2️⃣  Managing model stages (lifecycle)...")
    
    # Archive old production if exists
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if v.current_stage == "Production":
                print(f"   Archiving current Production (v{v.version})...")
                client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Archived"
                )
    except Exception as e:
        print(f"   No Production model to archive")
    
    # Move version 2 from Staging to Production
    try:
        print(f"   Promoting v2 to Production...")
        client.transition_model_version_stage(
            name=model_name,
            version=2,
            stage="Production"
        )
        print(f"   ✅ Version 2 → Production\n")
    except Exception as e:
        print(f"   ⚠️  Could not promote: {e}\n")
    
    # Move version 3 to Staging
    try:
        print(f"   Moving v3 to Staging for testing...")
        client.transition_model_version_stage(
            name=model_name,
            version=3,
            stage="Staging"
        )
        print(f"   ✅ Version 3 → Staging\n")
    except Exception as e:
        print(f"   ⚠️  Could not move to Staging: {e}\n")
    
    # ============================================================
    # 3. Add Model Version Descriptions
    # ============================================================
    print("3️⃣  Adding descriptions to model versions...")
    
    descriptions = {
        "1": "Initial balanced model with temp=0.7 - Good for general Q&A",
        "2": "Creative variant with temp=0.9 - Best for creative tasks",
        "3": "Precise variant with temp=0.3 - Optimal for factual queries"
    }
    
    for version, description in descriptions.items():
        try:
            client.update_model_version(
                name=model_name,
                version=version,
                description=description
            )
            print(f"   ✅ Updated description for v{version}")
        except Exception as e:
            print(f"   ⚠️  Could not update v{version}: {e}")
    
    print()
    
    # ============================================================
    # 4. Add Tags to Model Versions
    # ============================================================
    print("4️⃣  Adding tags to model versions...")
    
    version_tags = {
        "1": {"use_case": "general", "quality": "stable"},
        "2": {"use_case": "creative", "quality": "experimental"},
        "3": {"use_case": "factual", "quality": "stable"}
    }
    
    for version, tags in version_tags.items():
        for key, value in tags.items():
            try:
                client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key=key,
                    value=value
                )
            except Exception as e:
                pass
    
    print(f"   ✅ Added tags to all versions\n")
    
    # ============================================================
    # 5. View Updated Model Registry
    # ============================================================
    print("5️⃣  Updated model registry:")
    
    print_model_versions(client, model_name)
    
    # ============================================================
    # 6. Search Models by Tags
    # ============================================================
    print("6️⃣  Searching models by criteria...")
    
    # Search for stable models
    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        
        stable_versions = []
        for v in all_versions:
            tags = client.get_model_version(model_name, v.version).tags
            if tags.get("quality") == "stable":
                stable_versions.append(v.version)
        
        print(f"   Stable versions: {stable_versions}")
    except Exception as e:
        print(f"   ⚠️  Search error: {e}")
    
    # Search by use case
    try:
        creative_versions = []
        for v in all_versions:
            tags = client.get_model_version(model_name, v.version).tags
            if tags.get("use_case") == "creative":
                creative_versions.append(v.version)
        
        print(f"   Creative versions: {creative_versions}\n")
    except Exception as e:
        print(f"   ⚠️  Search error: {e}\n")
    
    # ============================================================
    # 7. Delete a Model Version
    # ============================================================
    print("7️⃣  Model version deletion (example - not executed)...")
    print("   To delete a version:")
    print(f"   client.delete_model_version(name='{model_name}', version='1')")
    print("   ⚠️  Use with caution - this is irreversible!\n")
    
    # ============================================================
    # 8. Rollback Scenario
    # ============================================================
    print("8️⃣  Rollback scenario (if Production model has issues)...")
    
    print("   Current state:")
    print("      Production: v2 (creative)")
    print("      Staging: v3 (precise)")
    print()
    print("   To rollback Production to v1:")
    print("      1. Move v2 from Production to Archived")
    print("      2. Move v1 from Archived to Production")
    print()
    print("   This allows safe rollback while preserving all versions\n")
    
    # ============================================================
    # Summary
    # ============================================================
    print("="*70)
    print("✨ Model Versioning Summary")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"\nLifecycle stages demonstrated:")
    print(f"  • None → Staging → Production → Archived")
    print(f"\nBest practices:")
    print(f"  ✅ Always test in Staging before Production")
    print(f"  ✅ Add descriptions and tags for documentation")
    print(f"  ✅ Archive old versions instead of deleting")
    print(f"  ✅ Keep rollback path available")
    print(f"\n💡 View in MLflow UI: {Config.MLFLOW_TRACKING_URI}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

