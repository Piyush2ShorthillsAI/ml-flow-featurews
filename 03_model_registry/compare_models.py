"""
Example: Model Comparison Framework
Side-by-side comparison of different models with detailed metrics
"""
import sys
sys.path.append('..')

import mlflow
import mlflow.pyfunc
from config import Config
import pandas as pd
from typing import Dict, List
import time


def create_test_scenarios():
    """Create diverse test scenarios for model comparison"""
    return pd.DataFrame({
        "scenario": [
            "Technical Definition",
            "Simple Explanation", 
            "Comparison Question",
            "How-To Question",
            "Application Question"
        ],
        "inputs": [
            "Define machine learning in technical terms",
            "Explain AI to a 10-year-old",
            "What's the difference between supervised and unsupervised learning?",
            "How do you train a neural network?",
            "What are real-world applications of computer vision?"
        ],
        "expected_style": [
            "Technical, precise, formal",
            "Simple, creative, engaging",
            "Balanced, comparative, structured",
            "Step-by-step, instructional",
            "Practical, example-rich"
        ]
    })


def evaluate_response_quality(response: str, expected_style: str) -> Dict:
    """Quick quality metrics for a response"""
    word_count = len(response.split())
    sentence_count = len([s for s in response.split('.') if s.strip()])
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # Style matching (simplified)
    style_keywords = {
        "technical": ["algorithm", "data", "model", "training", "optimization"],
        "simple": ["like", "imagine", "think", "easy", "simply"],
        "comparative": ["while", "whereas", "compared", "difference", "both"],
        "instructional": ["first", "then", "next", "step", "finally"],
        "practical": ["example", "such as", "application", "used", "real"]
    }
    
    style_matches = 0
    expected_lower = expected_style.lower()
    for style_type, keywords in style_keywords.items():
        if style_type in expected_lower:
            style_matches += sum(1 for kw in keywords if kw in response.lower())
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "style_matches": style_matches
    }


def compare_model_versions(versions: List[int], test_data: pd.DataFrame):
    """Compare multiple model versions side by side"""
    
    print(f"\n{'='*70}")
    print("Model Version Comparison")
    print(f"{'='*70}\n")
    
    model_name = "gemini_qa_model"
    results = {}
    
    # ============================================================
    # 1. Load All Models
    # ============================================================
    print(f"1️⃣  Loading {len(versions)} model versions...")
    
    models = {}
    for version in versions:
        try:
            model_uri = f"models:/{model_name}/{version}"
            models[version] = mlflow.pyfunc.load_model(model_uri)
            print(f"   ✅ Loaded version {version}")
        except Exception as e:
            print(f"   ⚠️  Could not load version {version}: {e}")
    
    print()
    
    if not models:
        print("❌ No models loaded. Run register_genai_model.py first!")
        return
    
    # ============================================================
    # 2. Run Predictions for All Scenarios
    # ============================================================
    print(f"2️⃣  Running predictions across {len(test_data)} scenarios...\n")
    
    comparison_results = []
    
    for idx, row in test_data.iterrows():
        scenario = row['scenario']
        question = row['inputs']
        expected_style = row['expected_style']
        
        print(f"   Scenario: {scenario}")
        print(f"   Question: {question[:60]}...")
        
        scenario_results = {
            'scenario': scenario,
            'question': question,
            'expected_style': expected_style
        }
        
        for version in sorted(models.keys()):
            try:
                start_time = time.time()
                response = models[version].predict({"question": question})
                elapsed_time = time.time() - start_time
                
                quality_metrics = evaluate_response_quality(response, expected_style)
                
                scenario_results[f'v{version}_response'] = response
                scenario_results[f'v{version}_time'] = round(elapsed_time, 3)
                scenario_results[f'v{version}_word_count'] = quality_metrics['word_count']
                scenario_results[f'v{version}_style_matches'] = quality_metrics['style_matches']
                
                print(f"      v{version}: {len(response)} chars, {elapsed_time:.2f}s")
                
            except Exception as e:
                print(f"      v{version}: ❌ Error - {e}")
                scenario_results[f'v{version}_response'] = f"ERROR: {e}"
        
        comparison_results.append(scenario_results)
        print()
    
    # ============================================================
    # 3. Detailed Comparison Analysis
    # ============================================================
    print(f"{'='*70}")
    print("📊 Detailed Comparison Results")
    print(f"{'='*70}\n")
    
    for result in comparison_results:
        print(f"Scenario: {result['scenario']}")
        print(f"Question: {result['question']}")
        print(f"Expected Style: {result['expected_style']}\n")
        
        for version in sorted(models.keys()):
            response_key = f'v{version}_response'
            time_key = f'v{version}_time'
            word_key = f'v{version}_word_count'
            style_key = f'v{version}_style_matches'
            
            if response_key in result:
                print(f"  Version {version}:")
                print(f"    Response: {result[response_key][:100]}...")
                if time_key in result:
                    print(f"    Time: {result[time_key]}s | Words: {result[word_key]} | Style matches: {result[style_key]}")
                print()
        
        print("-" * 70 + "\n")
    
    # ============================================================
    # 4. Aggregate Statistics
    # ============================================================
    print(f"{'='*70}")
    print("📈 Aggregate Statistics")
    print(f"{'='*70}\n")
    
    stats_df = pd.DataFrame(comparison_results)
    
    for version in sorted(models.keys()):
        print(f"Version {version}:")
        
        time_col = f'v{version}_time'
        word_col = f'v{version}_word_count'
        style_col = f'v{version}_style_matches'
        
        if time_col in stats_df.columns:
            avg_time = stats_df[time_col].mean()
            avg_words = stats_df[word_col].mean()
            avg_style = stats_df[style_col].mean()
            
            print(f"  Avg Response Time: {avg_time:.3f}s")
            print(f"  Avg Word Count: {avg_words:.1f}")
            print(f"  Avg Style Matches: {avg_style:.1f}")
            print()
    
    # ============================================================
    # 5. Save Comparison Report
    # ============================================================
    print("5️⃣  Logging comparison to MLflow...")
    
    with mlflow.start_run(run_name="model_comparison"):
        mlflow.set_tag("comparison_type", "version_comparison")
        mlflow.set_tag("models_compared", str(versions))
        mlflow.log_param("num_models", len(models))
        mlflow.log_param("num_scenarios", len(test_data))
        
        # Log aggregate metrics
        for version in sorted(models.keys()):
            time_col = f'v{version}_time'
            word_col = f'v{version}_word_count'
            style_col = f'v{version}_style_matches'
            
            if time_col in stats_df.columns:
                mlflow.log_metric(f"v{version}_avg_time", stats_df[time_col].mean())
                mlflow.log_metric(f"v{version}_avg_words", stats_df[word_col].mean())
                mlflow.log_metric(f"v{version}_avg_style_matches", stats_df[style_col].mean())
        
        # Save detailed results as artifact
        results_csv = "comparison_results.csv"
        stats_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(results_csv)
        
        print("✅ Comparison report logged to MLflow\n")
    
    return comparison_results


def main():
    """Main comparison workflow"""
    
    print("\n" + "="*70)
    print("Model Comparison Framework")
    print("="*70 + "\n")
    
    # Validate config
    if not Config.GEMINI_API_KEY:
        print("⚠️  ERROR: GEMINI_API_KEY not set")
        return
    
    Config.setup_mlflow()
    
    # Create test scenarios
    print("Creating test scenarios...")
    test_data = create_test_scenarios()
    print(f"✅ Created {len(test_data)} test scenarios\n")
    
    # Compare versions 1, 2, 3
    versions_to_compare = [1, 2, 3]
    results = compare_model_versions(versions_to_compare, test_data)
    
    # ============================================================
    # Summary
    # ============================================================
    print(f"{'='*70}")
    print("✨ Comparison Summary")
    print(f"{'='*70}")
    print(f"✅ Compared {len(versions_to_compare)} model versions")
    print(f"✅ Tested across {len(test_data)} diverse scenarios")
    print(f"✅ Evaluated response quality, speed, and style matching")
    print(f"\n💡 Use comparison results to:")
    print(f"   • Select best model for specific use cases")
    print(f"   • Understand version tradeoffs (speed vs quality)")
    print(f"   • Make informed deployment decisions")
    print(f"\n💡 View results in MLflow UI: {Config.MLFLOW_TRACKING_URI}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

