# 🚀 Quick Start with Gemini 2.5 Pro

This guide shows you how to get started with MLflow using **Google Gemini 2.5 Pro** - a powerful, cost-effective LLM with a massive context window.

## Why Gemini 2.5 Pro?

✅ **Large Context Window**: Up to 2M tokens (vs OpenAI's 128K)  
✅ **Cost-Effective**: Great performance at lower cost  
✅ **Advanced Reasoning**: Excellent for complex tasks  
✅ **Multimodal**: Supports text, images, video, audio  
✅ **Fast**: Optimized for production workloads  

## 🎯 5-Minute Setup

### 1. Get Your Gemini API Key (2 minutes)

1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your key

### 2. Configure the Project (1 minute)

```bash
cd /home/shtlp_0170/Videos/ml_flow_features

# Copy environment template
cp .env.example .env

# Edit .env file
nano .env  # or use your preferred editor
```

Add your Gemini API key:
```bash
GEMINI_API_KEY=your-actual-api-key-here
DEFAULT_LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-pro
```

### 3. Start MLflow Server (1 minute)

Open a **new terminal**:
```bash
cd /home/shtlp_0170/Videos/ml_flow_features
mlflow ui --port 5000
```

Keep this running. Access UI at: http://localhost:5000

### 4. Run Gemini Quick Start (1 minute)

In your original terminal:
```bash
python quickstart_gemini.py
```

🎉 **Done!** View results at http://localhost:5000

## 📚 Gemini-Specific Examples

### Basic Gemini Tracing
```bash
python 05_tracing/gemini_tracing.py
```

This demonstrates:
- ✅ Automatic tracing of Gemini calls
- ✅ Token usage tracking
- ✅ Multi-turn conversations
- ✅ Cost monitoring

### Example Output:
```
✅ Trace captured: abc123
   Token Usage:
     Input tokens: 245
     Output tokens: 156
     Total tokens: 401
```

## 💻 Sample Code

### Simple Gemini Call with Tracing

```python
import mlflow
import google.genai as genai
from config import Config

# Setup
Config.setup_mlflow()
mlflow.gemini.autolog()  # Enable auto-tracing

# Create client
client = genai.Client(api_key=Config.GEMINI_API_KEY)

# Generate content (automatically traced!)
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Explain MLflow in one sentence"
)

print(response.text)

# Get trace
trace_id = mlflow.get_last_active_trace_id()
trace = mlflow.get_trace(trace_id)
print(f"Tokens used: {trace.info.token_usage['total_tokens']}")
```

### Multi-Turn Chat with Gemini

```python
# Create chat session
chat = client.chats.create(model="gemini-2.5-pro")

# First turn
response1 = chat.send_message("What is machine learning?")
print(response1.text)

# Follow-up (context maintained)
response2 = chat.send_message("Give me a simple example")
print(response2.text)

# Both turns are traced automatically!
```

### Register Gemini-Optimized Prompt

```python
prompt = mlflow.genai.register_prompt(
    name="gemini_analysis_prompt",
    template="""You are an expert analyst using Gemini 2.5 Pro.
    
    Context: {{context}}
    Question: {{question}}
    
    Provide a detailed analysis using your extended context window.""",
    tags={"model": "gemini-2.5-pro", "type": "analysis"}
)
```

## 🎨 Gemini 2.5 Pro Features

### 1. Massive Context Window
```python
# Process large documents (up to 2M tokens!)
large_document = open("very_large_file.txt").read()  # Can be huge!

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=f"Analyze this document: {large_document}"
)
```

### 2. Advanced Reasoning
```python
# Complex analysis tasks
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="""Analyze this code for:
    1. Security vulnerabilities
    2. Performance issues
    3. Best practice violations
    
    [Your code here...]
    """
)
```

### 3. Multimodal Capabilities
```python
# Process images, video, audio alongside text
# (See Gemini docs for multimodal examples)
```

## 📊 Cost Comparison

### Gemini 2.5 Pro vs OpenAI GPT-4

| Feature | Gemini 2.5 Pro | GPT-4o |
|---------|----------------|---------|
| Context Window | 2M tokens | 128K tokens |
| Input Cost | $3.50/1M tokens | $5.00/1M tokens |
| Output Cost | $10.50/1M tokens | $15.00/1M tokens |
| Speed | Fast | Fast |

**Result**: Gemini is more cost-effective for large context tasks!

## 🔄 All Examples Work with Gemini

Most examples work with both OpenAI and Gemini. The project automatically uses your configured provider.

### Examples that work with Gemini:
- ✅ Prompt Registry (all examples)
- ✅ Prompt Evaluation
- ✅ Model Registry
- ✅ Custom Scorers
- ✅ Tracing (gemini_tracing.py)
- ✅ Evaluation Datasets
- ✅ End-to-End Pipeline

### To use Gemini everywhere:
Set in your `.env`:
```bash
DEFAULT_LLM_PROVIDER=gemini
```

## 🎯 Common Use Cases

### 1. Document Analysis
```bash
# Long-form document processing
python examples/document_analysis_gemini.py
```

### 2. Code Review
```bash
# Large codebase analysis
python examples/code_review_gemini.py
```

### 3. Research Assistant
```bash
# Multi-document synthesis
python examples/research_assistant_gemini.py
```

## 🐛 Troubleshooting

### Error: "GEMINI_API_KEY not set"
**Solution**: Make sure your `.env` file has:
```bash
GEMINI_API_KEY=your-actual-key
```

### Error: "Module 'google.genai' not found"
**Solution**: Install the package:
```bash
pip install google-genai
```

### Error: "Invalid API key"
**Solution**: 
1. Check your key at https://aistudio.google.com/app/apikey
2. Make sure there are no extra spaces in `.env`
3. Regenerate the key if needed

### Slow responses?
**Tip**: Gemini 2.5 Pro is optimized for large contexts. For short queries, standard Gemini might be faster.

## 📈 Monitoring & Cost Control

### Track Token Usage
```python
# After running evaluation
traces = mlflow.search_traces(max_results=10)

total_tokens = sum(
    trace.info.token_usage.get('total_tokens', 0)
    for trace in traces
    if hasattr(trace.info, 'token_usage')
)

print(f"Total tokens used: {total_tokens}")
print(f"Estimated cost: ${total_tokens * 0.0000035:.4f}")
```

### Set Budget Alerts
Monitor your usage in Google Cloud Console to set budget alerts.

## 🚀 Next Steps

1. **Run the examples**:
   ```bash
   python quickstart_gemini.py
   python 05_tracing/gemini_tracing.py
   ```

2. **Explore advanced features**:
   - Large context processing
   - Multimodal inputs
   - Function calling

3. **Build your application**:
   - Use the examples as templates
   - Customize prompts for your use case
   - Track everything in MLflow

## 📚 Resources

- **Gemini API Docs**: https://ai.google.dev/docs
- **Get API Key**: https://aistudio.google.com/app/apikey
- **MLflow Docs**: https://mlflow.org/docs/latest/
- **Pricing**: https://ai.google.dev/pricing

## 💡 Pro Tips

1. **Use the full context**: Gemini 2.5 Pro shines with large inputs
2. **Enable tracing**: Always use `mlflow.gemini.autolog()`
3. **Monitor costs**: Track token usage in MLflow
4. **Batch requests**: Process multiple items efficiently
5. **Cache prompts**: Use prompt registry for reusability

---

**Ready to build amazing GenAI apps with Gemini 2.5 Pro and MLflow! 🎉**

Need help? Check the main `README.md` or open an issue.

