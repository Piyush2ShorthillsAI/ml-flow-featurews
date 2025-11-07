"""
Example: Simple ResponsesAgent
Basic conversational agent with OpenAI-compatible interface
"""
import sys
sys.path.append('..')

import mlflow
from config import Config
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from mlflow.entities import SpanType
from openai import OpenAI


class SimpleQAAgent(ResponsesAgent):
    """Simple Q&A agent using OpenAI"""
    
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Generate response for the request"""
        # Extract messages from request
        messages = [msg.model_dump() for msg in request.input]
        
        # Call OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        # Create response
        return ResponsesAgentResponse(
            output=[
                self.create_text_output_item(
                    text=response.choices[0].message.content,
                    id="msg_1"
                )
            ]
        )


def main():
    print("\n" + "="*70)
    print("ResponsesAgent - Simple Agent")
    print("="*70 + "\n")
    
    if not Config.OPENAI_API_KEY:
        print("⚠️  ERROR: OPENAI_API_KEY not set")
        return
    
    Config.setup_mlflow()
    mlflow.openai.autolog()
    
    print("1️⃣  Creating agent...")
    agent = SimpleQAAgent()
    print("✅ Agent created\n")
    
    print("2️⃣  Testing agent...")
    
    # Create request
    request = ResponsesAgentRequest(
        input=[
            {"role": "user", "content": "What is MLflow?"}
        ]
    )
    
    # Get response
    response = agent.predict(request)
    
    print(f"✅ Response received:")
    print(f"   {response.output[0]['content'][0]['text'][:150]}...\n")
    
    print("3️⃣  Logging agent to MLflow...")
    
    with mlflow.start_run(run_name="simple_agent"):
        mlflow.pyfunc.log_model(
            artifact_path="agent",
            python_model=agent,
            registered_model_name="simple_qa_agent"
        )
        print("✅ Agent logged to MLflow\n")
    
    print("="*70)
    print("✨ ResponsesAgent provides:")
    print("   • OpenAI-compatible interface")
    print("   • Built-in tracing")
    print("   • Easy deployment")
    print(f"\n💡 View in MLflow UI: {Config.MLFLOW_TRACKING_URI}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

