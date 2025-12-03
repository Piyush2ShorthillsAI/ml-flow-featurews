"""
Example: Comprehensive Model Evaluation
Demonstrates evaluating models with multiple metrics and scorers
"""
import sys
sys.path.append('..')

import mlflow
import mlflow.pyfunc
from config import Config
from mlflow.genai import scorer
from mlflow.entities import Feedback
from mlflow.genai.judges import make_judge
import pandas as pd


def create_evaluation_dataset():
    """Create a comprehensive evaluation dataset"""
    return pd.DataFrame({
        "inputs": [
            {"question": "What is machine learning?"},
            {"question": "Explain neural networks"},
            {"question": "What is the difference between AI and ML?"},
            {"question": "How does deep learning work?"},
            {"question": "What are the applications of NLP?"}
        ],
        "ground_truth": [
            "Machine learning is a subset of AI that enables systems to learn from data.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "AI is the broader concept, ML is a subset of AI focused on learning from data.",
            "Deep learning uses multiple layers of neural networks to learn hierarchical representations.",
            "NLP applications include chatbots, translation, sentiment analysis, and text generation."
        ]
    })


# Custom Scorers for Model Evaluation
@scorer
def response_completeness(outputs: str, ground_truth: str) -> Feedback:
    """Measure how complete the response is compared to ground truth"""
    output_words = set(outputs.lower().split())
    truth_words = set(ground_truth.lower().split())
    
    # Remove common stop words
    stop_words = {'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of'}
    output_words -= stop_words
    truth_words -= stop_words
    
    if not truth_words:
        return Feedback(value=1.0, rationale="No ground truth to compare")
    
    overlap = output_words.intersection(truth_words)
    score = len(overlap) / len(truth_words)
    
    return Feedback(
        value=score,
        rationale=f"Covered {len(overlap)}/{len(truth_words)} key terms from ground truth",
        metadata={
            "overlap_count": len(overlap),
            "truth_terms": len(truth_words),
            "coverage_pct": round(score * 100, 2)
        }
    )


@scorer
def response_length_score(outputs: str) -> Feedback:
    """Evaluate response length appropriateness"""
    word_count = len(outputs.split())
    
    if 20 <= word_count <= 150:
        score = 1.0
        rationale = f"Optimal length: {word_count} words"
    elif word_count < 20:
        score = word_count / 20
        rationale = f"Too brief: {word_count} words (optimal: 20-150)"
    else:
        score = max(0.3, 1 - (word_count - 150) / 200)
        rationale = f"Too verbose: {word_count} words (optimal: 20-150)"
    
    return Feedback(
        value=score,
        rationale=rationale,
        metadata={"word_count": word_count}
    )


@scorer
def clarity_score(outputs: str) -> Feedback:
    """Evaluate response clarity"""
    if not outputs.strip():
        return Feedback(value=0, rationale="Empty response")
    
    sentences = [s.strip() for s in outputs.split('.') if s.strip()]
    words = outputs.split()
    
    if not sentences:
        return Feedback(value=0.5, rationale="No clear sentences")
    
    # Average sentence length
    avg_length = len(words) / len(sentences)
    
    # Optimal: 12-25 words per sentence
    if 12 <= avg_length <= 25:
        length_score = 1.0
    elif avg_length < 12:
        length_score = 0.6 + (avg_length / 12) * 0.4
    else:
        length_score = max(0.4, 1 - (avg_length - 25) / 40)
    
    return Feedback(
        value=length_score,
        rationale=f"Average sentence length: {avg_length:.1f} words",
        metadata={
            "sentence_count": len(sentences),
            "avg_sentence_length": round(avg_length, 2)
        }
    )


@scorer
def technical_accuracy(outputs: str, ground_truth: str) -> Feedback:
    """Check for technical terms alignment"""
    # Extract key technical terms
    tech_terms_truth = [word for word in ground_truth.lower().split() 
                        if len(word) > 5 and word.isalnum()]
    tech_terms_output = [word for word in outputs.lower().split() 
                         if len(word) > 5 and word.isalnum()]
    
    if not tech_terms_truth:
        return Feedback(value=1.0, rationale="No technical terms to verify")
    
    matches = sum(1 for term in tech_terms_truth if term in ' '.join(tech_terms_output))
    score = matches / len(tech_terms_truth)
    
    return Feedback(
        value=score,
        rationale=f"Matched {matches}/{len(tech_terms_truth)} technical terms",
        metadata={"matched_terms": matches, "total_terms": len(tech_terms_truth)}
    )


def evaluate_model_version(model_uri: str, eval_data: pd.DataFrame, model_name: str):
    """Evaluate a specific model version"""
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"Model URI: {model_uri}")
    print(f"{'='*70}\n")
    
    try:
        # Load model
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Create prediction function
        # Parameter name must match the key in inputs dict: "question"
        def predict_fn(question: str) -> str:
            return model.predict({"question": question})
        
        # Create LLM judges for additional evaluation
        relevance_judge = make_judge(
            name="relevance",
            instructions="Rate the relevance of {{ outputs }} to {{ inputs }}. Use scale: highly_relevant/relevant/somewhat_relevant/not_relevant",
            model="gemini:/gemini-1.5-flash"
        )
        
        helpfulness_judge = make_judge(
            name="helpfulness",
            instructions="Rate how helpful {{ outputs }} is for answering {{ inputs }}. Use scale: very_helpful/helpful/somewhat_helpful/not_helpful",
            model="gemini:/gemini-1.5-flash"
        )
        
        # Combine custom scorers and LLM judges
        all_scorers = [
            response_completeness,
            response_length_score,
            clarity_score,
            technical_accuracy,
            relevance_judge,
            helpfulness_judge
        ]
        
        print(f"Running evaluation with {len(all_scorers)} scorers...")
        print(f"  • {len([s for s in all_scorers if hasattr(s, '__wrapped__')])} custom scorers")
        print(f"  • {len([s for s in all_scorers if not hasattr(s, '__wrapped__')])} LLM judges\n")
        
        # Run evaluation
        with mlflow.start_run(run_name=f"evaluate_{model_name}"):
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("model_uri", model_uri)
            mlflow.set_tag("evaluation_type", "comprehensive")
            
            results = mlflow.genai.evaluate(
                data=eval_data,
                predict_fn=predict_fn,
                scorers=all_scorers
            )
            
            # Log additional metrics
            metrics = results.metrics
            
            print(f"\n{'='*70}")
            print(f"📊 Evaluation Results for {model_name}")
            print(f"{'='*70}\n")
            
            print("Aggregate Metrics:")
            for metric_name, value in sorted(metrics.items()):
                if isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.3f}")
            
            print(f"\n{'='*70}\n")
            
            return results
            
    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}\n")
        return None


def main():
    """Main evaluation workflow"""
    
    print("\n" + "="*70)
    print("Comprehensive Model Evaluation")
    print("="*70 + "\n")
    
    # Validate config
    if not Config.GEMINI_API_KEY:
        print("⚠️  ERROR: GEMINI_API_KEY not set")
        return
    
    Config.setup_mlflow()
    
    # Create evaluation dataset
    print("1️⃣  Creating evaluation dataset...")
    eval_data = create_evaluation_dataset()
    print(f"✅ Created dataset with {len(eval_data)} examples\n")
    
    # ============================================================
    # 2. Evaluate All Model Versions
    # ============================================================
    print("2️⃣  Evaluating all registered model versions...")
    
    model_name = "gemini_qa_model"
    versions_to_evaluate = [1, 2, 3]
    
    results_summary = {}
    
    for version in versions_to_evaluate:
        model_uri = f"models:/{model_name}/{version}"
        result = evaluate_model_version(
            model_uri=model_uri,
            eval_data=eval_data,
            model_name=f"{model_name}_v{version}"
        )
        
        if result:
            results_summary[f"v{version}"] = result.metrics
    
    # ============================================================
    # 3. Compare Results Across Versions
    # ============================================================
    if results_summary:
        print(f"\n{'='*70}")
        print("📊 Model Version Comparison")
        print(f"{'='*70}\n")
        
        # Get all metric names
        all_metrics = set()
        for metrics in results_summary.values():
            all_metrics.update(metrics.keys())
        
        # Sort metrics
        sorted_metrics = sorted(all_metrics)
        
        # Print comparison table
        print(f"{'Metric':<40} " + " ".join([f"{'v'+v[1]:<12}" for v in results_summary.keys()]))
        print("-" * 70)
        
        for metric in sorted_metrics:
            if all(isinstance(results_summary[v].get(metric), (int, float)) for v in results_summary):
                values = [results_summary[v].get(metric, 0) for v in results_summary]
                metric_str = f"{metric:<40}"
                values_str = " ".join([f"{v:<12.3f}" for v in values])
                print(metric_str + values_str)
        
        print()
    
    # ============================================================
    # 4. Recommendations
    # ============================================================
    print(f"{'='*70}")
    print("💡 Recommendations")
    print(f"{'='*70}")
    print("✅ Evaluated models using multiple metrics:")
    print("   • Custom code-based scorers")
    print("   • LLM-as-a-judge evaluators")
    print("   • Ground truth comparison")
    print("\n📈 Use these results to:")
    print("   • Identify best-performing model version")
    print("   • Understand model strengths and weaknesses")
    print("   • Make data-driven promotion decisions")
    print("   • Track performance trends over time")
    print(f"\n💡 View detailed results in MLflow UI: {Config.MLFLOW_TRACKING_URI}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

