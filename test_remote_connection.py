#!/usr/bin/env python3
"""
Test Remote MLflow Server Connection
Verifies that you can connect to https://mlflow.shorthills.ai
"""
import sys
sys.path.append('.')

from config import Config
import mlflow


def test_connection():
    """Test connection to remote MLflow server"""
    
    print("\n" + "="*70)
    print("Testing Remote MLflow Server Connection")
    print("="*70 + "\n")
    
    # Check configuration
    print("1️⃣  Checking configuration...")
    print(f"   Tracking URI: {Config.MLFLOW_TRACKING_URI}")
    print(f"   Experiment Name: {Config.MLFLOW_EXPERIMENT_NAME}")
    
    if not Config.MLFLOW_TRACKING_USERNAME or not Config.MLFLOW_TRACKING_PASSWORD:
        print("\n❌ ERROR: Authentication not configured!")
        print("\nPlease set in your .env file:")
        print("   MLFLOW_TRACKING_USERNAME=your_username")
        print("   MLFLOW_TRACKING_PASSWORD=your_password")
        return False
    
    print(f"   Username: {Config.MLFLOW_TRACKING_USERNAME}")
    print(f"   Password: {'*' * len(Config.MLFLOW_TRACKING_PASSWORD)}")
    print()
    
    # Setup MLflow
    print("2️⃣  Connecting to remote server...")
    try:
        Config.setup_mlflow()
        print("   ✅ MLflow configured\n")
    except Exception as e:
        print(f"   ❌ Configuration failed: {e}\n")
        return False
    
    # Test connection by getting/creating experiment
    print("3️⃣  Testing experiment access...")
    try:
        exp = mlflow.get_experiment_by_name(Config.MLFLOW_EXPERIMENT_NAME)
        if exp:
            print(f"   ✅ Found experiment: {Config.MLFLOW_EXPERIMENT_NAME}")
            print(f"   Experiment ID: {exp.experiment_id}")
        else:
            print(f"   Creating new experiment: {Config.MLFLOW_EXPERIMENT_NAME}")
            exp_id = mlflow.create_experiment(Config.MLFLOW_EXPERIMENT_NAME)
            print(f"   ✅ Created experiment ID: {exp_id}")
        print()
    except Exception as e:
        print(f"   ❌ Experiment access failed: {e}\n")
        print("\nPossible issues:")
        print("   • Check username/password are correct")
        print("   • Verify you have access to the server")
        print("   • Check internet connection")
        return False
    
    # Test creating a run
    print("4️⃣  Testing run creation...")
    try:
        with mlflow.start_run(run_name="connection_test") as run:
            # Log test data
            mlflow.log_param("test_param", "remote_connection")
            mlflow.log_metric("test_metric", 1.0)
            mlflow.set_tag("test_type", "connection_verification")
            
            print(f"   ✅ Created test run: {run.info.run_id}")
            print(f"   Run Name: connection_test")
        print()
    except Exception as e:
        print(f"   ❌ Run creation failed: {e}\n")
        return False
    
    # Test searching runs
    print("5️⃣  Testing data retrieval...")
    try:
        runs = mlflow.search_runs(
            experiment_names=[Config.MLFLOW_EXPERIMENT_NAME],
            max_results=5
        )
        print(f"   ✅ Retrieved runs: {len(runs)} found")
        if len(runs) > 0:
            print(f"   Latest run: {runs.iloc[0]['tags.mlflow.runName']}")
        print()
    except Exception as e:
        print(f"   ⚠️  Data retrieval warning: {e}\n")
    
    # Success!
    print("="*70)
    print("✅ CONNECTION TEST SUCCESSFUL!")
    print("="*70)
    print("\nYour configuration is working correctly!")
    print(f"\n🌐 View your results at: {Config.MLFLOW_TRACKING_URI}")
    print(f"📊 Experiment: {Config.MLFLOW_EXPERIMENT_NAME}")
    
    print("\n💡 Next Steps:")
    print("   1. Run: python quickstart_gemini.py")
    print("   2. View results on the remote server")
    print("   3. Explore other examples")
    
    print("\n⚠️  Important Reminders:")
    print("   • Use unique names for experiments/prompts/models")
    print("   • Include your name/identifier in all names")
    print("   • Don't modify others' experiments")
    print("   • Results are visible to your team")
    print("="*70 + "\n")
    
    return True


def main():
    try:
        success = test_connection()
        if not success:
            print("\n" + "="*70)
            print("❌ CONNECTION TEST FAILED")
            print("="*70)
            print("\nTroubleshooting steps:")
            print("   1. Check .env file has correct credentials")
            print("   2. Verify server URL is correct")
            print("   3. Test internet connection")
            print("   4. Contact MLflow admin if issues persist")
            print("="*70 + "\n")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

