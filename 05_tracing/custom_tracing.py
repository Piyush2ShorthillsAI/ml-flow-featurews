"""
Example: Custom Tracing with @mlflow.trace
Demonstrates manual tracing for custom functions
"""
import sys
sys.path.append('..')

import mlflow
from config import Config
from openai import OpenAI
import time


@mlflow.trace
def retrieve_context(question: str) -> str:
    """Simulated context retrieval"""
    time.sleep(0.1)  # Simulate retrieval time
    knowledge_base = {
        "mlflow": "MLflow is an open-source platform for ML lifecycle management",
        "ai": "AI is intelligence demonstrated by machines",
        "python": "Python is a high-level programming language"
    }
    
    for key, value in knowledge_base.items():
        if key in question.lower():
            return value
    return "General knowledge context"


@mlflow.trace(span_type="LLM")
def generate_response(question: str, context: str) -> str:
    """Generate response using LLM"""
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


@mlflow.trace(name="rag_pipeline", span_type="CHAIN")
def rag_pipeline(question: str) -> str:
    """Complete RAG pipeline with tracing"""
    # Retrieve context (traced)
    context = retrieve_context(question)
    
    # Generate response (traced)
    response = generate_response(question, context)
    
    return response


def main():
    print("\n" + "="*70)
    print("Custom Tracing with @mlflow.trace")
    print("="*70 + "\n")
    
    if not Config.OPENAI_API_KEY:
        print("⚠️  ERROR: OPENAI_API_KEY not set")
        return
    
    Config.setup_mlflow()
    mlflow.openai.autolog()
    
    print("Running RAG pipeline with custom tracing...\n")
    
    # Execute pipeline
    question = "What is MLflow?"
    result = rag_pipeline(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result[:150]}...\n")
    
    # Get trace details
    trace_id = mlflow.get_last_active_trace_id()
    if trace_id:
        trace = mlflow.get_trace(trace_id)
        print(f"✅ Trace captured: {trace_id}")
        print(f"   Spans: {len(trace.data.spans)}")
        
        print(f"\n   Span hierarchy:")
        for span in trace.data.spans:
            print(f"   - {span.name} ({span.span_type})")
    
    print("\n" + "="*70)
    print("✨ Custom tracing enables:")
    print("   • End-to-end pipeline visibility")
    print("   • Performance debugging")
    print("   • Component-level monitoring")
    print(f"\n💡 View in MLflow UI: {Config.MLFLOW_TRACKING_URI}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

