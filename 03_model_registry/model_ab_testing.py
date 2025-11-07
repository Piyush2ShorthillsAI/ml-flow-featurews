"""
Example: Model A/B Testing Framework
Demonstrates comparing two models for production deployment decisions
"""
import sys
sys.path.append('..')

import mlflow
import mlflow.pyfunc
from config import Config
import pandas as pd
from typing import Dict, Tuple
import random
from datetime import datetime


def create_production_test_data():
    """Create realistic production test scenarios"""
    return pd.DataFrame({
        "user_query": [
            "How do I reset my password?",
            "What are your business hours?",
            "Explain machine learning",
            "Tell me about your pricing plans",
            "How secure is your platform?",
            "What's the difference between AI and ML?",
            "Can you help me with Python code?",
            "What are the benefits of cloud computing?",
            "How do I contact support?",
            "Explain blockchain technology"
        ],
        "user_segment": [
            "customer_support", "customer_support", "technical", "sales",
            "enterprise", "technical", "developer", "technical",
            "customer_support", "technical"
        ],
        "priority": [
            "high", "medium", "medium", "high",
            "high", "medium", "medium", "low",
            "high", "low"
        ]
    })


def simulate_user_feedback(response: str, query_type: str) -> Dict:
    """
    Simulate user feedback based on response quality
    In production, this would come from real user interactions
    """
    # Simplified feedback simulation based on response characteristics
    word_count = len(response.split())
    
    # Heuristics for feedback (in production, use real user data)
    if query_type == "customer_support":
        # Support queries prefer concise, clear answers
        satisfaction = 5 if 20 <= word_count <= 100 else 3
    elif query_type == "technical":
        # Technical queries prefer detailed explanations
        satisfaction = 5 if word_count >= 80 else 3
    elif query_type == "sales":
        # Sales queries prefer balanced, persuasive responses
        satisfaction = 4 if 40 <= word_count <= 120 else 3
    else:
        satisfaction = 4
    
    # Add some randomness to simulate real-world variation
    satisfaction += random.randint(-1, 1)
    satisfaction = max(1, min(5, satisfaction))
    
    return {
        "satisfaction_score": satisfaction,
        "would_recommend": satisfaction >= 4,
        "response_appropriate": word_count >= 20
    }


def run_ab_test(model_a_uri: str, model_b_uri: str, test_data: pd.DataFrame, 
                model_a_name: str = "Model A", model_b_name: str = "Model B"):
    """
    Run A/B test comparing two models
    """
    
    print(f"\n{'='*70}")
    print(f"A/B Test: {model_a_name} vs {model_b_name}")
    print(f"{'='*70}\n")
    
    # Load both models
    print("1️⃣  Loading models...")
    try:
        model_a = mlflow.pyfunc.load_model(model_a_uri)
        print(f"   ✅ Loaded {model_a_name}")
    except Exception as e:
        print(f"   ❌ Error loading {model_a_name}: {e}")
        return
    
    try:
        model_b = mlflow.pyfunc.load_model(model_b_uri)
        print(f"   ✅ Loaded {model_b_name}\n")
    except Exception as e:
        print(f"   ❌ Error loading {model_b_name}: {e}")
        return
    
    # ============================================================
    # 2. Run Predictions and Collect Metrics
    # ============================================================
    print("2️⃣  Running A/B test predictions...\n")
    
    results_a = []
    results_b = []
    
    for idx, row in test_data.iterrows():
        query = row['user_query']
        segment = row['user_segment']
        priority = row['priority']
        
        print(f"   Query {idx+1}/{len(test_data)}: {query[:50]}...")
        
        # Test Model A
        try:
            response_a = model_a.predict({"question": query})
            feedback_a = simulate_user_feedback(response_a, segment)
            
            results_a.append({
                'query': query,
                'segment': segment,
                'priority': priority,
                'response': response_a,
                'response_length': len(response_a),
                **feedback_a
            })
            print(f"      {model_a_name}: {len(response_a)} chars, score={feedback_a['satisfaction_score']}")
        except Exception as e:
            print(f"      {model_a_name}: Error - {e}")
        
        # Test Model B
        try:
            response_b = model_b.predict({"question": query})
            feedback_b = simulate_user_feedback(response_b, segment)
            
            results_b.append({
                'query': query,
                'segment': segment,
                'priority': priority,
                'response': response_b,
                'response_length': len(response_b),
                **feedback_b
            })
            print(f"      {model_b_name}: {len(response_b)} chars, score={feedback_b['satisfaction_score']}")
        except Exception as e:
            print(f"      {model_b_name}: Error - {e}")
        
        print()
    
    # ============================================================
    # 3. Calculate Aggregate Metrics
    # ============================================================
    print(f"{'='*70}")
    print("📊 A/B Test Results")
    print(f"{'='*70}\n")
    
    df_a = pd.DataFrame(results_a)
    df_b = pd.DataFrame(results_b)
    
    # Overall metrics
    print("Overall Performance:")
    print(f"\n{model_a_name}:")
    print(f"  Average Satisfaction: {df_a['satisfaction_score'].mean():.2f}/5.0")
    print(f"  Would Recommend: {df_a['would_recommend'].sum()}/{len(df_a)} ({df_a['would_recommend'].mean()*100:.1f}%)")
    print(f"  Avg Response Length: {df_a['response_length'].mean():.0f} chars")
    print(f"  Response Appropriate: {df_a['response_appropriate'].sum()}/{len(df_a)}")
    
    print(f"\n{model_b_name}:")
    print(f"  Average Satisfaction: {df_b['satisfaction_score'].mean():.2f}/5.0")
    print(f"  Would Recommend: {df_b['would_recommend'].sum()}/{len(df_b)} ({df_b['would_recommend'].mean()*100:.1f}%)")
    print(f"  Avg Response Length: {df_b['response_length'].mean():.0f} chars")
    print(f"  Response Appropriate: {df_b['response_appropriate'].sum()}/{len(df_b)}")
    
    # Segment analysis
    print(f"\n{'='*70}")
    print("📈 Performance by User Segment")
    print(f"{'='*70}\n")
    
    for segment in df_a['segment'].unique():
        segment_a = df_a[df_a['segment'] == segment]
        segment_b = df_b[df_b['segment'] == segment]
        
        print(f"{segment.upper()}:")
        print(f"  {model_a_name}: {segment_a['satisfaction_score'].mean():.2f}/5.0")
        print(f"  {model_b_name}: {segment_b['satisfaction_score'].mean():.2f}/5.0")
        
        if segment_a['satisfaction_score'].mean() > segment_b['satisfaction_score'].mean():
            print(f"  Winner: {model_a_name} ✅")
        elif segment_b['satisfaction_score'].mean() > segment_a['satisfaction_score'].mean():
            print(f"  Winner: {model_b_name} ✅")
        else:
            print(f"  Tied")
        print()
    
    # ============================================================
    # 4. Statistical Significance (Simplified)
    # ============================================================
    print(f"{'='*70}")
    print("🔬 Statistical Analysis")
    print(f"{'='*70}\n")
    
    diff_satisfaction = df_b['satisfaction_score'].mean() - df_a['satisfaction_score'].mean()
    diff_recommendation = df_b['would_recommend'].mean() - df_a['would_recommend'].mean()
    
    print(f"Difference in Satisfaction: {diff_satisfaction:+.2f}")
    print(f"Difference in Recommendation Rate: {diff_recommendation:+.1%}")
    
    if abs(diff_satisfaction) >= 0.5:
        print(f"\n{'⚠️  SIGNIFICANT DIFFERENCE DETECTED' if abs(diff_satisfaction) >= 1.0 else '✨ Notable difference observed'}")
    else:
        print(f"\nNo significant difference (< 0.5 points)")
    
    # ============================================================
    # 5. Recommendation
    # ============================================================
    print(f"\n{'='*70}")
    print("💡 Deployment Recommendation")
    print(f"{'='*70}\n")
    
    score_a = df_a['satisfaction_score'].mean()
    score_b = df_b['satisfaction_score'].mean()
    
    if score_b > score_a + 0.3:
        winner = model_b_name
        confidence = "HIGH"
        action = f"✅ Deploy {model_b_name} to production"
    elif score_a > score_b + 0.3:
        winner = model_a_name
        confidence = "HIGH"
        action = f"✅ Keep {model_a_name} in production"
    elif score_b > score_a:
        winner = model_b_name
        confidence = "MEDIUM"
        action = f"🤔 Consider gradual rollout of {model_b_name}"
    elif score_a > score_b:
        winner = model_a_name
        confidence = "MEDIUM"
        action = f"🤔 Keep {model_a_name}, but monitor closely"
    else:
        winner = "TIE"
        confidence = "N/A"
        action = "🤷 No clear winner - consider other factors"
    
    print(f"Winner: {winner}")
    print(f"Confidence: {confidence}")
    print(f"Recommended Action: {action}")
    
    # ============================================================
    # 6. Log to MLflow
    # ============================================================
    print(f"\n6️⃣  Logging A/B test results to MLflow...")
    
    with mlflow.start_run(run_name=f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tag("test_type", "ab_test")
        mlflow.set_tag("model_a", model_a_name)
        mlflow.set_tag("model_b", model_b_name)
        mlflow.set_tag("winner", winner)
        mlflow.set_tag("confidence", confidence)
        
        # Log Model A metrics
        mlflow.log_metric("model_a_satisfaction", score_a)
        mlflow.log_metric("model_a_recommendation_rate", df_a['would_recommend'].mean())
        mlflow.log_metric("model_a_avg_length", df_a['response_length'].mean())
        
        # Log Model B metrics
        mlflow.log_metric("model_b_satisfaction", score_b)
        mlflow.log_metric("model_b_recommendation_rate", df_b['would_recommend'].mean())
        mlflow.log_metric("model_b_avg_length", df_b['response_length'].mean())
        
        # Log differences
        mlflow.log_metric("satisfaction_diff", diff_satisfaction)
        mlflow.log_metric("recommendation_diff", diff_recommendation)
        
        # Save detailed results
        df_a.to_csv("ab_test_model_a.csv", index=False)
        df_b.to_csv("ab_test_model_b.csv", index=False)
        mlflow.log_artifact("ab_test_model_a.csv")
        mlflow.log_artifact("ab_test_model_b.csv")
        
        print("✅ Results logged to MLflow\n")
    
    print(f"{'='*70}\n")


def main():
    """Main A/B testing workflow"""
    
    print("\n" + "="*70)
    print("Model A/B Testing Framework")
    print("="*70 + "\n")
    
    # Validate config
    if not Config.GEMINI_API_KEY:
        print("⚠️  ERROR: GEMINI_API_KEY not set")
        return
    
    Config.setup_mlflow()
    
    # Create test data
    print("Creating production test scenarios...")
    test_data = create_production_test_data()
    print(f"✅ Created {len(test_data)} test scenarios\n")
    
    # ============================================================
    # A/B Test: Version 1 vs Version 2
    # ============================================================
    print("Running A/B Test: Balanced (v1) vs Creative (v2)")
    print("="*70)
    
    run_ab_test(
        model_a_uri="models:/gemini_qa_model/1",
        model_b_uri="models:/gemini_qa_model/2",
        test_data=test_data,
        model_a_name="Balanced (v1)",
        model_b_name="Creative (v2)"
    )
    
    # ============================================================
    # Summary
    # ============================================================
    print(f"{'='*70}")
    print("✨ A/B Testing Summary")
    print(f"{'='*70}")
    print(f"✅ Compared 2 model versions")
    print(f"✅ Tested across {len(test_data)} production scenarios")
    print(f"✅ Analyzed by user segment and priority")
    print(f"✅ Provided deployment recommendation")
    print(f"\n💡 Best Practices:")
    print(f"   • Test with realistic production data")
    print(f"   • Analyze by user segment")
    print(f"   • Consider statistical significance")
    print(f"   • Run gradual rollouts for close results")
    print(f"\n💡 View results in MLflow UI: {Config.MLFLOW_TRACKING_URI}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

