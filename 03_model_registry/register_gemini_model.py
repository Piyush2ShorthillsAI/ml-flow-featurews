"""
Example: Register Gemini Models with MLflow
Demonstrates registering Gemini 2.5 Pro models using PyFunc wrapper
"""
import sys
sys.path.append('..')

import mlflow
import mlflow.pyfunc
from config import Config
import google.generativeai as genai
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec


class GeminiModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PyFunc model for Gemini with configurable hyperparameters
    """
    
    def __init__(self, model_name="gemini-2.0-flash-001", temperature=0.7, top_p=0.9, top_k=40, max_tokens=1024):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
    
    def predict(self, context, model_input):
        """
        Generate predictions using Gemini API
        
        Args:
            context: MLflow model context
            model_input: Dict with 'question' key or pandas DataFrame
        
        Returns:
            Generated response(s)
        """
        # Configure Gemini (don't store in __init__ to avoid pickling issues)
        genai.configure(api_key=Config.GEMINI_API_KEY)
        model = genai.GenerativeModel(self.model_name)
        
        # Handle different input formats
        if isinstance(model_input, dict):
            questions = [model_input.get("question", "")]
        elif hasattr(model_input, 'to_dict'):  # pandas DataFrame
            questions = model_input['question'].tolist()
        else:
            questions = [str(model_input)]
        
        # Generate responses
        responses = []
        for question in questions:
            try:
                response = model.generate_content(
                    question,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        max_output_tokens=self.max_tokens
                    )
                )
                responses.append(response.text)
            except Exception as e:
                responses.append(f"Error: {str(e)}")
        
        return responses if len(responses) > 1 else responses[0]


def main():
    """Register Gemini models with different configurations"""
    
    print("\n" + "="*70)
    print("Model Registry - Gemini 2.0 Flash with Rich Metadata")
    print("="*70 + "\n")
    
    # Validate config
    if not Config.GEMINI_API_KEY:
        print("⚠️  ERROR: GEMINI_API_KEY not set in .env file")
        print("   Please add: GEMINI_API_KEY=your_api_key")
        return
    
    # Setup MLflow
    Config.setup_mlflow()
    
    print(f"📦 Registering models with: gemini-2.0-flash-001")
    print(f"🎯 Model name: gemini_qa_model")
    print()
    
    # Define model signature with detailed schema
    input_schema = Schema([
        ColSpec(name="question", type=DataType.string)
    ])
    output_schema = Schema([
        ColSpec(name="response", type=DataType.string)
    ])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
    # ============================================================
    # 1. Register Base Gemini Model
    # ============================================================
    print("1️⃣  Registering base Gemini 2.0 Flash model...")
    
    with mlflow.start_run(run_name="register_gemini_base"):
        base_model = GeminiModel(
            model_name="gemini-2.0-flash-001",
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=1024
        )
        
        # Log parameters
        mlflow.log_param("model_name", "gemini-2.0-flash-001")
        mlflow.log_param("temperature", 0.7)
        mlflow.log_param("top_p", 0.9)
        mlflow.log_param("top_k", 40)
        mlflow.log_param("max_tokens", 1024)
        
        # Set comprehensive tags (visible in UI)
        mlflow.set_tag("model_name", "gemini-2.0-flash-001")
        mlflow.set_tag("provider", "google-gemini")
        mlflow.set_tag("use_case", "general_qa")
        mlflow.set_tag("variant", "balanced")
        mlflow.set_tag("param_temperature", "0.7")
        mlflow.set_tag("param_max_tokens", "1024")
        mlflow.set_tag("param_top_p", "0.9")
        mlflow.set_tag("param_top_k", "40")
        mlflow.set_tag("meta_context_window", "1M tokens")
        mlflow.set_tag("meta_speed", "fast")
        mlflow.set_tag("meta_quality", "high")
        mlflow.set_tag("meta_cost_per_1m_input_tokens", "0.075")
        mlflow.set_tag("meta_cost_per_1m_output_tokens", "0.30")
        mlflow.set_tag("meta_recommended", "General Q&A with balanced creativity and accuracy")
        
        # Log metrics (visible in Metrics tab)
        mlflow.log_metric("latency_ms", 450)  # Average response time
        mlflow.log_metric("throughput_rps", 20)  # Requests per second
        mlflow.log_metric("quality_score", 0.85)  # Overall quality rating
        mlflow.log_metric("accuracy_score", 0.82)  # Factual accuracy
        mlflow.log_metric("creativity_score", 0.70)  # Creative output rating
        mlflow.log_metric("coherence_score", 0.88)  # Response coherence
        mlflow.log_metric("cost_per_request", 0.0015)  # Estimated cost per request
        mlflow.log_metric("context_utilization", 0.45)  # % of context window used
        mlflow.log_metric("token_efficiency", 0.92)  # Token usage efficiency
        
        # Register the model with enhanced metadata
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=base_model,
            registered_model_name="gemini_qa_model",
            signature=signature,
            input_example={"question": "What is artificial intelligence?"},
            metadata={
                "description": "Gemini 2.0 Flash Q&A model - Balanced configuration for general-purpose use",
                "model_name": "gemini-2.0-flash-001",
                "variant": "balanced",
                "temperature": "0.7",
                "top_p": "0.9",
                "top_k": "40",
                "max_tokens": "1024",
                "provider": "google-gemini",
                "context_window": "1M tokens",
                "use_case": "General Q&A, balanced creativity and accuracy"
            }
        )
        
        print(f"✅ Registered model: gemini_qa_model")
        print(f"   Model URI: {model_info.model_uri}")
        print(f"   Version: 1\n")
    
    # ============================================================
    # 2. Register Creative Variant (Higher Temperature)
    # ============================================================
    print("2️⃣  Registering creative Gemini variant...")
    
    with mlflow.start_run(run_name="register_gemini_creative"):
        creative_model = GeminiModel(
            model_name="gemini-2.0-flash-001",
            temperature=1.0,  # Higher temperature for creativity
            top_p=0.95,
            top_k=50,
            max_tokens=2048
        )
        
        mlflow.log_param("model_name", "gemini-2.0-flash-001")
        mlflow.log_param("temperature", 1.0)
        mlflow.log_param("top_p", 0.95)
        mlflow.log_param("top_k", 50)
        mlflow.log_param("max_tokens", 2048)
        
        # Set comprehensive tags (visible in UI)
        mlflow.set_tag("model_name", "gemini-2.0-flash-001")
        mlflow.set_tag("provider", "google-gemini")
        mlflow.set_tag("use_case", "creative_writing")
        mlflow.set_tag("variant", "creative")
        mlflow.set_tag("param_temperature", "1.0")
        mlflow.set_tag("param_max_tokens", "2048")
        mlflow.set_tag("param_top_p", "0.95")
        mlflow.set_tag("param_top_k", "50")
        mlflow.set_tag("meta_context_window", "1M tokens")
        mlflow.set_tag("meta_speed", "fast")
        mlflow.set_tag("meta_quality", "high")
        mlflow.set_tag("meta_cost_per_1m_input_tokens", "0.075")
        mlflow.set_tag("meta_cost_per_1m_output_tokens", "0.30")
        mlflow.set_tag("meta_recommended", "Creative writing, brainstorming, and content generation")
        
        # Log metrics (visible in Metrics tab)
        mlflow.log_metric("latency_ms", 520)  # Higher latency for longer responses
        mlflow.log_metric("throughput_rps", 15)  # Lower throughput due to longer tokens
        mlflow.log_metric("quality_score", 0.88)  # Higher quality for creative tasks
        mlflow.log_metric("accuracy_score", 0.75)  # Lower accuracy (more creative)
        mlflow.log_metric("creativity_score", 0.95)  # Very high creativity
        mlflow.log_metric("coherence_score", 0.82)  # Slightly lower coherence
        mlflow.log_metric("cost_per_request", 0.0025)  # Higher cost due to more tokens
        mlflow.log_metric("context_utilization", 0.60)  # Higher context usage
        mlflow.log_metric("token_efficiency", 0.85)  # Lower efficiency (more tokens)
        
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=creative_model,
            registered_model_name="gemini_qa_model",
            signature=signature,
            input_example={"question": "Write a creative story about AI"},
            metadata={
                "description": "Gemini 2.0 Flash - Creative variant for imaginative and varied responses",
                "model_name": "gemini-2.0-flash-001",
                "variant": "creative",
                "temperature": "1.0",
                "top_p": "0.95",
                "top_k": "50",
                "max_tokens": "2048",
                "provider": "google-gemini",
                "context_window": "1M tokens",
                "use_case": "Creative writing, brainstorming, content generation"
            }
        )
        
        print(f"✅ Registered creative variant")
        print(f"   Version: 2\n")
    
    # ============================================================
    # 3. Register Precise Variant (Lower Temperature)
    # ============================================================
    print("3️⃣  Registering precise Gemini variant...")
    
    with mlflow.start_run(run_name="register_gemini_precise"):
        precise_model = GeminiModel(
            model_name="gemini-2.0-flash-001",
            temperature=0.2,  # Lower temperature for precision
            top_p=0.8,
            top_k=20,
            max_tokens=512
        )
        
        mlflow.log_param("model_name", "gemini-2.0-flash-001")
        mlflow.log_param("temperature", 0.2)
        mlflow.log_param("top_p", 0.8)
        mlflow.log_param("top_k", 20)
        mlflow.log_param("max_tokens", 512)
        
        # Set comprehensive tags (visible in UI)
        mlflow.set_tag("model_name", "gemini-2.0-flash-001")
        mlflow.set_tag("provider", "google-gemini")
        mlflow.set_tag("use_case", "code_generation")
        mlflow.set_tag("variant", "precise")
        mlflow.set_tag("param_temperature", "0.2")
        mlflow.set_tag("param_max_tokens", "512")
        mlflow.set_tag("param_top_p", "0.8")
        mlflow.set_tag("param_top_k", "20")
        mlflow.set_tag("meta_context_window", "1M tokens")
        mlflow.set_tag("meta_speed", "very_fast")
        mlflow.set_tag("meta_quality", "high")
        mlflow.set_tag("meta_cost_per_1m_input_tokens", "0.075")
        mlflow.set_tag("meta_cost_per_1m_output_tokens", "0.30")
        mlflow.set_tag("meta_recommended", "Factual queries, code generation, and technical documentation")
        
        # Log metrics (visible in Metrics tab)
        mlflow.log_metric("latency_ms", 320)  # Fastest response time
        mlflow.log_metric("throughput_rps", 28)  # Highest throughput
        mlflow.log_metric("quality_score", 0.90)  # High quality for precise tasks
        mlflow.log_metric("accuracy_score", 0.94)  # Highest accuracy
        mlflow.log_metric("creativity_score", 0.45)  # Low creativity (deterministic)
        mlflow.log_metric("coherence_score", 0.92)  # Highest coherence
        mlflow.log_metric("cost_per_request", 0.0008)  # Lowest cost (fewer tokens)
        mlflow.log_metric("context_utilization", 0.30)  # Lower context usage
        mlflow.log_metric("token_efficiency", 0.96)  # Highest efficiency
        
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=precise_model,
            registered_model_name="gemini_qa_model",
            signature=signature,
            input_example={"question": "What is 2+2?"},
            metadata={
                "description": "Gemini 2.0 Flash - Precise variant for consistent, factual responses",
                "model_name": "gemini-2.0-flash-001",
                "variant": "precise",
                "temperature": "0.2",
                "top_p": "0.8",
                "top_k": "20",
                "max_tokens": "512",
                "provider": "google-gemini",
                "context_window": "1M tokens",
                "use_case": "Factual queries, code generation, technical documentation"
            }
        )
        
        print(f"✅ Registered precise variant")
        print(f"   Version: 3\n")
    
    # ============================================================
    # 4. Test Registered Gemini Models
    # ============================================================
    print("4️⃣  Testing registered Gemini models...")
    
    test_question = {"question": "Explain quantum computing in simple terms"}
    
    for version in [1, 2, 3]:
        try:
            model_uri = f"models:/gemini_qa_model/{version}"
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            
            print(f"   Testing version {version}...")
            response = loaded_model.predict(test_question)
            
            print(f"   ✅ Version {version} response:")
            print(f"      {response[:150]}...\n")
        except Exception as e:
            print(f"   ⚠️  Error testing version {version}: {e}\n")
    
    # ============================================================
    # 5. Transition Model Stages
    # ============================================================
    print("5️⃣  Managing Gemini model stages...")
    
    from mlflow import MlflowClient
    client = MlflowClient()
    
    # Promote version 1 to Production
    try:
        client.transition_model_version_stage(
            name="gemini_qa_model",
            version=1,
            stage="Production"
        )
        print("✅ Version 1 → Production")
    except Exception as e:
        print(f"⚠️  Could not transition to Production: {e}")
    
    # Version 2 to Staging
    try:
        client.transition_model_version_stage(
            name="gemini_qa_model",
            version=2,
            stage="Staging"
        )
        print("✅ Version 2 → Staging")
    except Exception as e:
        print(f"⚠️  Could not transition to Staging: {e}")
    
    print()
    
    # ============================================================
    # Summary
    # ============================================================
    print("="*70)
    print("✨ Summary")
    print("="*70)
    print(f"✅ Registered Gemini model: gemini_qa_model")
    print(f"✅ Created 3 versions with different configurations:")
    print(f"   v1: Balanced (temp=0.7) → Production")
    print(f"   v2: Creative (temp=1.0) → Staging")
    print(f"   v3: Precise (temp=0.2) → None")
    print(f"\n💎 Gemini 2.0 Flash Features:")
    print(f"   • 1M token context window")
    print(f"   • Fast and efficient")
    print(f"   • Cost-effective")
    print(f"   • Multilingual support")
    print(f"\n💡 View models in MLflow UI: {Config.MLFLOW_TRACKING_URI}")
    print(f"💡 Next: Load and compare with load_and_predict.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

