"""
Example: Load and Use Registered Models
Demonstrates loading models by version, stage, and making predictions
"""
import sys
sys.path.append('..')

import mlflow
import mlflow.pyfunc
from config import Config
import pandas as pd


def main():
    """Load and use registered models"""
    
    print("\n" + "="*70)
    print("Model Registry - Load and Predict")
    print("="*70 + "\n")
    
    # Validate config
    if not Config.GEMINI_API_KEY:
        print("⚠️  ERROR: GEMINI_API_KEY not set")
        return
    
    # Setup MLflow
    Config.setup_mlflow()
    
    # ============================================================
    # 1. Load Model by Version
    # ============================================================
    print("1️⃣  Loading model by specific version...")
    
    try:
        model_v1 = mlflow.pyfunc.load_model("models:/gemini_qa_model/1")
        print(f"✅ Loaded gemini_qa_model version 1")
        
        # Test prediction
        response = model_v1.predict({"question": "What is MLflow?"})
        print(f"   Response: {response[:100]}...\n")
    except Exception as e:
        print(f"⚠️  Error: {e}")
        print(f"   Run ../03_model_registry/register_gemini_model.py first!\n")
    
    # ============================================================
    # 2. Load Model by Stage
    # ============================================================
    print("2️⃣  Loading model by stage...")
    
    try:
        # Load production model
        prod_model = mlflow.pyfunc.load_model("models:/gemini_qa_model/Production")
        print(f"✅ Loaded Production model")
        
        response = prod_model.predict({"question": "Explain machine learning"})
        print(f"   Response: {response[:100]}...\n")
    except Exception as e:
        print(f"⚠️  Could not load Production model: {e}\n")
    
    try:
        # Load staging model
        staging_model = mlflow.pyfunc.load_model("models:/gemini_qa_model/Staging")
        print(f"✅ Loaded Staging model")
        
        response = staging_model.predict({"question": "What is deep learning?"})
        print(f"   Response: {response[:100]}...\n")
    except Exception as e:
        print(f"⚠️  Could not load Staging model: {e}\n")
    
    # ============================================================
    # 3. Batch Predictions with DataFrame
    # ============================================================
    print("3️⃣  Making batch predictions...")
    
    try:
        model = mlflow.pyfunc.load_model("models:/gemini_qa_model/1")
        
        # Create batch of questions
        batch_data = pd.DataFrame({
            "question": [
                "What is artificial intelligence?",
                "What is machine learning?",
                "What is deep learning?"
            ]
        })
        
        print(f"✅ Processing batch of {len(batch_data)} questions...")
        responses = model.predict(batch_data)
        
        print(f"\nBatch results:")
        for i, (q, r) in enumerate(zip(batch_data['question'], responses), 1):
            print(f"\n   Q{i}: {q}")
            print(f"   A{i}: {r[:80]}...\n")
    except Exception as e:
        print(f"⚠️  Error in batch prediction: {e}\n")
    
    # ============================================================
    # 4. Compare Different Model Versions
    # ============================================================
    print("4️⃣  Comparing different model versions...")
    
    test_question = {"question": "Explain neural networks creatively"}
    
    print(f"Question: {test_question['question']}\n")
    
    for version in [1, 2, 3]:
        try:
            model = mlflow.pyfunc.load_model(f"models:/gemini_qa_model/{version}")
            response = model.predict(test_question)
            
            print(f"Version {version} response:")
            print(f"  {response[:120]}...")
            print()
        except Exception as e:
            print(f"Version {version}: ⚠️  {e}\n")
    
    # ============================================================
    # 5. Get Model Metadata
    # ============================================================
    print("5️⃣  Retrieving model metadata...")
    
    from mlflow import MlflowClient
    client = MlflowClient()
    
    try:
        # Get all versions of the model
        model_versions = client.search_model_versions("name='gemini_qa_model'")
        
        print(f"✅ Found {len(model_versions)} versions:\n")
        
        for mv in model_versions:
            print(f"   Version {mv.version}:")
            print(f"      Stage: {mv.current_stage}")
            print(f"      Status: {mv.status}")
            if mv.run_id:
                run = client.get_run(mv.run_id)
                print(f"      Temperature: {run.data.params.get('temperature', 'N/A')}")
                print(f"      Variant: {run.data.tags.get('variant', 'base')}")
            print()
    except Exception as e:
        print(f"⚠️  Error getting metadata: {e}\n")
    
    # ============================================================
    # 6. Load Latest Version
    # ============================================================
    print("6️⃣  Loading latest version...")
    
    try:
        # Get latest version number
        versions = client.search_model_versions("name='gemini_qa_model'")
        if versions:
            latest_version = max([int(v.version) for v in versions])
            
            latest_model = mlflow.pyfunc.load_model(f"models:/gemini_qa_model/{latest_version}")
            print(f"✅ Loaded latest version: {latest_version}")
            
            response = latest_model.predict({"question": "What is the latest version?"})
            print(f"   Response: {response[:100]}...\n")
    except Exception as e:
        print(f"⚠️  Error: {e}\n")
    
    # ============================================================
    # Summary
    # ============================================================
    print("="*70)
    print("✨ Summary")
    print("="*70)
    print(f"✅ Loaded models by version number")
    print(f"✅ Loaded models by stage (Production, Staging)")
    print(f"✅ Made single and batch predictions")
    print(f"✅ Compared responses across versions")
    print(f"✅ Retrieved model metadata")
    print(f"\n💡 Models can be deployed using MLflow serving")
    print(f"💡 Next: See model versioning in model_versioning.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

