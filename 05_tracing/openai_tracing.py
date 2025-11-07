"""
Example: OpenAI Tracing with MLflow
Demonstrates automatic tracing of OpenAI API calls
"""
import sys
sys.path.append('..')

import mlflow
from config import Config
from openai import OpenAI


def main():
    print("\n" + "="*70)
    print("OpenAI Tracing with MLflow")
    print("="*70 + "\n")
    
    if not Config.OPENAI_API_KEY:
        print("⚠️  ERROR: OPENAI_API_KEY not set")
        return
    
    Config.setup_mlflow()
    
    # Enable auto-tracing for OpenAI
    mlflow.openai.autolog()
    print("✅ Enabled OpenAI auto-tracing\n")
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    # Single completion
    print("1️⃣  Single completion (traced automatically)...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is MLflow?"}
        ],
        temperature=0.7
    )
    print(f"   Response: {response.choices[0].message.content[:100]}...\n")
    
    # Get trace
    trace_id = mlflow.get_last_active_trace_id()
    if trace_id:
        trace = mlflow.get_trace(trace_id)
        print(f"✅ Trace captured: {trace_id}")
        if hasattr(trace.info, 'token_usage'):
            usage = trace.info.token_usage
            print(f"   Tokens: {usage.get('total_tokens', 0)} total\n")
    
    # Multi-turn conversation
    print("2️⃣  Multi-turn conversation...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    
    messages.append({"role": "user", "content": "Can you give an example?"})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    print(f"✅ Multi-turn conversation traced\n")
    
    # Search traces
    print("3️⃣  Searching traces...")
    traces = mlflow.search_traces(max_results=5)
    print(f"✅ Found {len(traces)} recent traces\n")
    
    print("="*70)
    print("✨ Summary")
    print("="*70)
    print("✅ OpenAI calls automatically traced")
    print("✅ Token usage tracked")
    print("✅ Traces searchable in MLflow")
    print(f"\n💡 View traces in UI: {Config.MLFLOW_TRACKING_URI}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

