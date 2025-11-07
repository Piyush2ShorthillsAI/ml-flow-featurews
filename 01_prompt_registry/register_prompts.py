"""
Example: Register and Version Prompts in MLflow
Demonstrates how to register prompts, create versions, and manage prompt lifecycle
"""
import sys
sys.path.append('..')

import mlflow
from config import Config

def main():
    """Register and version prompts in MLflow Prompt Registry"""
    
    print("\n" + "="*70)
    print("MLflow Prompt Registry - Registration Example")
    print("="*70 + "\n")
    
    # Setup MLflow
    Config.setup_mlflow()
    
    # ============================================================
    # 1. Register a Simple Text Prompt
    # ============================================================
    print("1️⃣  Registering a simple text prompt...")
    
    simple_prompt = mlflow.genai.register_prompt(
        name="qa_prompt_simple",
        template="Answer the following question: {{question}}",
        commit_message="Initial version of simple Q&A prompt",
        tags={"type": "qa", "version": "v1"}
    )
    
    print(f"✅ Registered prompt: {simple_prompt.name}")
    print(f"   Version: {simple_prompt.version}")
    print(f"   Template: {simple_prompt.template}\n")
    
    # ============================================================
    # 2. Register a Chat-style Prompt (List format)
    # ============================================================
    print("2️⃣  Registering a chat-style prompt...")
    
    chat_template = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that answers questions clearly and concisely."
        },
        {
            "role": "user",
            "content": "Question: {{question}}"
        }
    ]
    
    chat_prompt = mlflow.genai.register_prompt(
        name="qa_prompt_chat",
        template=chat_template,
        commit_message="Chat-style Q&A prompt",
        tags={"type": "chat", "version": "v1"}
    )
    
    print(f"✅ Registered chat prompt: {chat_prompt.name}")
    print(f"   Version: {chat_prompt.version}\n")
    
    # ============================================================
    # 3. Create a New Version of Existing Prompt
    # ============================================================
    print("3️⃣  Creating version 2 of the prompt...")
    
    improved_template = [
        {
            "role": "system",
            "content": "You are a knowledgeable AI assistant. Provide detailed, accurate answers with examples when appropriate."
        },
        {
            "role": "user",
            "content": "Question: {{question}}\n\nPlease provide a comprehensive answer."
        }
    ]
    
    chat_prompt_v2 = mlflow.genai.register_prompt(
        name="qa_prompt_chat",  # Same name creates new version
        template=improved_template,
        commit_message="Improved prompt with more detailed instructions",
        tags={"type": "chat", "version": "v2"}
    )
    
    print(f"✅ Created new version: {chat_prompt_v2.version}")
    print(f"   Previous version: {chat_prompt.version}\n")
    
    # ============================================================
    # 4. Register Specialized Prompts
    # ============================================================
    print("4️⃣  Registering specialized prompts...")
    
    # Summarization prompt
    summarization_prompt = mlflow.genai.register_prompt(
        name="summarization_prompt",
        template="Summarize the following text in {{num_sentences}} sentences:\n\n{{text}}",
        commit_message="Text summarization prompt",
        tags={"task": "summarization"}
    )
    
    print(f"✅ Registered summarization prompt")
    
    # Classification prompt
    classification_prompt = mlflow.genai.register_prompt(
        name="sentiment_classifier",
        template="Classify the sentiment of this text as positive, negative, or neutral:\n\n{{text}}\n\nSentiment:",
        commit_message="Sentiment classification prompt",
        tags={"task": "classification"}
    )
    
    print(f"✅ Registered classification prompt")
    
    # Chain-of-thought prompt
    cot_prompt = mlflow.genai.register_prompt(
        name="cot_reasoning_prompt",
        template="Let's solve this step by step:\n\nQuestion: {{question}}\n\nThinking process:",
        commit_message="Chain-of-thought reasoning prompt",
        tags={"task": "reasoning", "technique": "cot"}
    )
    
    print(f"✅ Registered chain-of-thought prompt\n")
    
    # ============================================================
    # 5. Set Aliases for Prompt Versions
    # ============================================================
    print("5️⃣  Setting aliases for prompt versions...")
    
    # Note: 'latest' is reserved by MLflow, use 'champion' instead
    # Set 'champion' alias to version 2 (best performer)
    mlflow.genai.set_prompt_alias(
        name="qa_prompt_chat",
        alias="champion",
        version=chat_prompt_v2.version
    )
    print(f"✅ Set 'champion' alias to version {chat_prompt_v2.version}")
    
    # Set 'baseline' alias to version 1 (stable reference)
    mlflow.genai.set_prompt_alias(
        name="qa_prompt_chat",
        alias="baseline",
        version=chat_prompt.version
    )
    print(f"✅ Set 'baseline' alias to version {chat_prompt.version}")
    
    # Set 'challenger' alias to version 2 (testing)
    mlflow.genai.set_prompt_alias(
        name="qa_prompt_chat",
        alias="challenger",
        version=chat_prompt_v2.version
    )
    print(f"✅ Set 'challenger' alias to version {chat_prompt_v2.version}\n")
    
    # ============================================================
    # Summary
    # ============================================================
    print("="*70)
    print("✨ Summary")
    print("="*70)
    print(f"✅ Registered 5 different prompts")
    print(f"✅ Created 2 versions of 'qa_prompt_chat'")
    print(f"✅ Set 3 aliases (latest, production, staging)")
    print(f"\n💡 View prompts in MLflow UI: {Config.MLFLOW_TRACKING_URI}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

