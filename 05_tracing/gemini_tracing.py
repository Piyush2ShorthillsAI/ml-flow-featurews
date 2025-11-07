"""
Example: Gemini 2.5 Pro Tracing with MLflow
Demonstrates automatic tracing of Google Gemini API calls
"""
import sys
sys.path.append('..')

import mlflow
from config import Config
import google.genai as genai


def main():
    print("\n" + "="*70)
    print("Gemini 2.5 Pro Tracing with MLflow")
    print("="*70 + "\n")
    
    if not Config.GEMINI_API_KEY:
        print("⚠️  ERROR: GEMINI_API_KEY not set in .env file")
        print("Please add your Gemini API key and try again")
        return
    
    Config.setup_mlflow()
    
    # Enable auto-tracing for Gemini
    mlflow.gemini.autolog()
    print("✅ Enabled Gemini auto-tracing\n")
    
    # Configure Gemini client
    client = genai.Client(api_key=Config.GEMINI_API_KEY)
    
    # Single completion
    print("1️⃣  Single completion with Gemini 2.5 Pro (traced automatically)...")
    response = client.models.generate_content(
        model=Config.GEMINI_MODEL,
        contents="What is MLflow and why is it useful for GenAI applications?"
    )
    print(f"   Response: {response.text[:150]}...\n")
    
    # Get trace
    trace_id = mlflow.get_last_active_trace_id()
    if trace_id:
        trace = mlflow.get_trace(trace_id)
        print(f"✅ Trace captured: {trace_id}")
        if hasattr(trace.info, 'token_usage') and trace.info.token_usage:
            usage = trace.info.token_usage
            print(f"   Token Usage:")
            print(f"     Input tokens: {usage.get('input_tokens', 0)}")
            print(f"     Output tokens: {usage.get('output_tokens', 0)}")
            print(f"     Total tokens: {usage.get('total_tokens', 0)}")
        print()
    
    # Multi-turn chat
    print("2️⃣  Multi-turn conversation with Gemini...")
    chat = client.chats.create(model=Config.GEMINI_MODEL)
    
    response1 = chat.send_message("Explain machine learning in one sentence.")
    print(f"   User: Explain machine learning in one sentence.")
    print(f"   Gemini: {response1.text}\n")
    
    response2 = chat.send_message("Now explain it for a 10-year-old.")
    print(f"   User: Now explain it for a 10-year-old.")
    print(f"   Gemini: {response2.text}\n")
    
    print("✅ Multi-turn conversation traced\n")
    
    # Complex prompt with context
    print("3️⃣  Complex prompt with context...")
    complex_prompt = """
    You are an expert in MLflow and machine learning operations.
    
    Context: A data science team wants to track their GenAI experiments.
    
    Question: What are the top 3 features of MLflow they should use first?
    
    Provide a concise, numbered list with brief explanations.
    """
    
    response = client.models.generate_content(
        model=Config.GEMINI_MODEL,
        contents=complex_prompt
    )
    print(f"   Response:\n{response.text}\n")
    
    # Search traces
    print("4️⃣  Searching traces...")
    traces = mlflow.search_traces(max_results=5)
    print(f"✅ Found {len(traces)} recent traces")
    
    if traces:
        print(f"\n   Recent traces:")
        for i, trace in enumerate(traces[:3], 1):
            print(f"   {i}. Trace ID: {trace.info.trace_id}")
            if hasattr(trace.info, 'token_usage') and trace.info.token_usage:
                total = trace.info.token_usage.get('total_tokens', 0)
                print(f"      Tokens: {total}")
    print()
    
    # Token usage analysis
    print("5️⃣  Token usage analysis...")
    total_input_tokens = 0
    total_output_tokens = 0
    
    for trace in traces:
        if hasattr(trace.info, 'token_usage') and trace.info.token_usage:
            usage = trace.info.token_usage
            total_input_tokens += usage.get('input_tokens', 0)
            total_output_tokens += usage.get('output_tokens', 0)
    
    if total_input_tokens or total_output_tokens:
        print(f"   Total across all traces:")
        print(f"     Input tokens: {total_input_tokens}")
        print(f"     Output tokens: {total_output_tokens}")
        print(f"     Total tokens: {total_input_tokens + total_output_tokens}")
    print()
    
    print("="*70)
    print("✨ Summary")
    print("="*70)
    print("✅ Gemini 2.5 Pro calls automatically traced")
    print("✅ Token usage tracked for cost monitoring")
    print("✅ Multi-turn conversations captured")
    print("✅ Traces searchable and analyzable in MLflow")
    print(f"\n💡 View traces in UI: {Config.MLFLOW_TRACKING_URI}")
    print("💡 Gemini 2.5 Pro provides:")
    print("   • Large context window (up to 2M tokens)")
    print("   • Advanced reasoning capabilities")
    print("   • Multilingual support")
    print("   • Cost-effective for large prompts")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

