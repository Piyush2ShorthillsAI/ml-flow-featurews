"""
Example: Custom Code-Based Scorers
Demonstrates creating custom evaluation scorers using the @scorer decorator
"""
import sys
sys.path.append('..')

import mlflow
from config import Config
import google.generativeai as genai
from utils import create_sample_qa_data, print_evaluation_results

from mlflow.genai import scorer
from mlflow.entities import Feedback
import re

# Enable Gemini tracing
mlflow.gemini.autolog()


def main():
    """Demonstrate custom code-based scorers"""
    
    print("\n" + "="*70)
    print("Evaluation Framework - Custom Scorers")
    print("="*70 + "\n")
    
    # Validate and setup
    if not Config.GEMINI_API_KEY:
        print("⚠️  ERROR: GEMINI_API_KEY not set")
        return
    
    Config.setup_mlflow()
    genai.configure(api_key=Config.GEMINI_API_KEY)
    
    # ============================================================
    # 1. Simple Boolean Scorers
    # ============================================================
    print("1️⃣  Creating simple boolean scorers...")
    
    @scorer
    def is_not_empty(outputs: str) -> bool:
        """Check if output is not empty"""
        return bool(outputs and outputs.strip())
    
    @scorer
    def is_reasonable_length(outputs: str) -> bool:
        """Check if output length is reasonable (10-500 words)"""
        word_count = len(outputs.split())
        return 10 <= word_count <= 500
    
    @scorer
    def no_error_messages(outputs: str) -> bool:
        """Check for error messages in output"""
        error_keywords = ['error', 'exception', 'failed', 'could not']
        return not any(keyword in outputs.lower() for keyword in error_keywords)
    
    print("✅ Created 3 boolean scorers\n")
    
    # ============================================================
    # 2. Numeric Scorers with Feedback
    # ============================================================
    print("2️⃣  Creating numeric scorers with feedback...")
    
    @scorer
    def word_count_score(outputs: str) -> Feedback:
        """Score based on word count (optimal: 30-100 words)"""
        word_count = len(outputs.split())
        
        if 30 <= word_count <= 100:
            score = 1.0
            rationale = f"Optimal length: {word_count} words"
        elif word_count < 30:
            score = word_count / 30
            rationale = f"Too short: {word_count} words (optimal: 30-100)"
        else:
            score = max(0, 1 - (word_count - 100) / 200)
            rationale = f"Too long: {word_count} words (optimal: 30-100)"
        
        return Feedback(
            value=score,
            rationale=rationale,
            metadata={"word_count": word_count}
        )
    
    @scorer
    def readability_score(outputs: str) -> Feedback:
        """Simple readability score based on sentence structure"""
        sentences = [s.strip() for s in outputs.split('.') if s.strip()]
        words = outputs.split()
        
        if not sentences or not words:
            return Feedback(value=0, rationale="Empty output")
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Optimal: 15-20 words per sentence
        if 15 <= avg_sentence_length <= 20:
            score = 1.0
        elif avg_sentence_length < 15:
            score = 0.7 + (avg_sentence_length / 15) * 0.3
        else:
            score = max(0.3, 1 - (avg_sentence_length - 20) / 30)
        
        return Feedback(
            value=score,
            rationale=f"Average sentence length: {avg_sentence_length:.1f} words",
            metadata={
                "sentence_count": len(sentences),
                "avg_sentence_length": avg_sentence_length
            }
        )
    
    print("✅ Created 2 numeric scorers with feedback\n")
    
    # ============================================================
    # 3. Content-Based Scorers
    # ============================================================
    print("3️⃣  Creating content-based scorers...")
    
    @scorer
    def technical_terms_present(outputs: str) -> Feedback:
        """Check for technical terms in output"""
        technical_terms = [
            'algorithm', 'data', 'model', 'learning', 'neural',
            'training', 'prediction', 'classification', 'regression'
        ]
        
        found_terms = [term for term in technical_terms if term in outputs.lower()]
        score = min(len(found_terms) / 3, 1.0)  # Expect at least 3 terms
        
        return Feedback(
            value=score,
            rationale=f"Found {len(found_terms)} technical terms: {', '.join(found_terms[:5])}",
            metadata={"terms_found": found_terms}
        )
    
    @scorer
    def has_examples(outputs: str) -> Feedback:
        """Check if output includes examples"""
        example_indicators = [
            'for example', 'for instance', 'such as', 'e.g.',
            'like', 'including', 'namely'
        ]
        
        has_example = any(indicator in outputs.lower() for indicator in example_indicators)
        
        return Feedback(
            value=1.0 if has_example else 0.0,
            rationale="Includes examples" if has_example else "No examples provided"
        )
    
    @scorer
    def structured_response(outputs: str) -> Feedback:
        """Check if response is well-structured"""
        structure_score = 0
        features = []
        
        # Check for numbered lists
        if re.search(r'\d+\.', outputs):
            structure_score += 0.3
            features.append("numbered list")
        
        # Check for bullet points
        if re.search(r'[•\-\*]\s', outputs):
            structure_score += 0.3
            features.append("bullet points")
        
        # Check for paragraphs
        if outputs.count('\n\n') >= 1:
            structure_score += 0.2
            features.append("paragraphs")
        
        # Check for clear sections
        if any(marker in outputs for marker in [':', 'Definition', 'Example', 'Note']):
            structure_score += 0.2
            features.append("sections")
        
        return Feedback(
            value=min(structure_score, 1.0),
            rationale=f"Structure features: {', '.join(features) if features else 'plain text'}",
            metadata={"features": features}
        )
    
    print("✅ Created 3 content-based scorers\n")
    
    # ============================================================
    # 4. Expectation-Based Scorers
    # ============================================================
    print("4️⃣  Creating expectation-based scorers...")
    
    @scorer
    def key_concepts_coverage(outputs: str, expectations: dict) -> Feedback:
        """Check coverage of key concepts from expectations"""
        key_concepts = expectations.get("key_concepts", [])
        if not key_concepts:
            return Feedback(value=1.0, rationale="No key concepts specified")
        
        output_lower = outputs.lower()
        found_concepts = [c for c in key_concepts if c.lower() in output_lower]
        missing_concepts = [c for c in key_concepts if c.lower() not in output_lower]
        
        score = len(found_concepts) / len(key_concepts)
        
        return Feedback(
            value=score,
            rationale=f"Coverage: {len(found_concepts)}/{len(key_concepts)} concepts. Missing: {missing_concepts}",
            metadata={
                "total_concepts": len(key_concepts),
                "found_concepts": found_concepts,
                "missing_concepts": missing_concepts
            }
        )
    
    print("✅ Created expectation-based scorer\n")
    
    # ============================================================
    # 5. Run Evaluation with Custom Scorers
    # ============================================================
    print("5️⃣  Running evaluation with custom scorers...")
    
    @mlflow.trace
    def predict_fn(question: str) -> str:
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        system_prompt = "You are a helpful assistant that explains technical concepts clearly."
        full_prompt = f"{system_prompt}\n\n{question}"
        
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.7)
        )
        return response.text
    
    eval_data = create_sample_qa_data()
    
    all_scorers = [
        is_not_empty,
        is_reasonable_length,
        no_error_messages,
        word_count_score,
        readability_score,
        technical_terms_present,
        has_examples,
        structured_response,
        key_concepts_coverage
    ]
    
    print(f"   Running evaluation with {len(all_scorers)} custom scorers...")
    print("   This may take a moment...\n")
    
    with mlflow.start_run(run_name="custom_scorers_evaluation"):
        mlflow.set_tag("scorer_type", "custom_code_based")
        mlflow.log_param("num_scorers", len(all_scorers))
        
        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=all_scorers
        )
        
        print_evaluation_results(results, "Custom Scorers Evaluation")
    
    # ============================================================
    # Summary
    # ============================================================
    print("="*70)
    print("✨ Summary")
    print("="*70)
    print(f"✅ Created {len(all_scorers)} custom scorers:")
    print(f"   • Boolean scorers (pass/fail)")
    print(f"   • Numeric scorers with detailed feedback")
    print(f"   • Content analysis scorers")
    print(f"   • Expectation-based scorers")
    print(f"\n💡 Custom scorers provide full control over evaluation logic")
    print(f"💡 View results in MLflow UI: {Config.MLFLOW_TRACKING_URI}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

