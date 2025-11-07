# MLflow Features Demonstration Project

A comprehensive demonstration of MLflow 3.x features for GenAI applications, including prompt management, model tracking, evaluation frameworks, and agent serving.

## 🚀 Features Covered

This project demonstrates all major MLflow features:

### 1. **Prompt Registry & Versioning**
- Register and version prompts
- Load prompts with aliases (latest, production, staging)
- Track prompt evolution across experiments

### 2. **Prompt Evaluation**
- Evaluate prompts with custom scorers
- Compare prompt versions side-by-side
- Track evaluation metrics

### 3. **Model Registration & Versioning** ✨
- Register Gemini 2.5 Pro models using PyFunc wrapper
- 2M token context window support
- Version models with hyperparameters
- Track model metadata and lineage
- Comprehensive model evaluation (custom scorers + LLM judges)
- Side-by-side model comparison
- A/B testing framework for production
- Performance monitoring and drift detection

### 4. **Evaluation Framework**
- Custom code-based scorers
- LLM-as-a-Judge scorers (Guidelines, make_judge)
- Template-based scorers
- Versioning scorers

### 5. **Tracing & Observability**
- Auto-trace OpenAI, Anthropic, Gemini calls
- Custom tracing with @mlflow.trace
- Multi-turn conversation tracing
- Agent tool-calling traces

### 6. **Evaluation Datasets**
- Create datasets from traces
- Build datasets from dictionaries/DataFrames
- Add expectations and ground truth
- Version and share datasets

### 7. **ResponsesAgent**
- Build conversational agents
- OpenAI-compatible agent interface
- Tool-calling agents
- Streaming support

### 8. **Experiment Tracking**
- Log parameters, metrics, artifacts
- Track runs and experiments
- Search and compare runs
- Version control for experiments

### 9. **End-to-End Workflows**
- Evaluation-driven development
- Production monitoring patterns
- Complete GenAI application lifecycle

## 📁 Project Structure

```
ml_flow_features/
├── config.py                          # Configuration management
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── README.md                          # This file
│
├── 01_prompt_registry/                # Prompt management examples
│   ├── register_prompts.py            # Register and version prompts
│   ├── load_prompts.py                # Load prompts with aliases
│   └── prompt_lifecycle.py            # Manage prompt lifecycle
│
├── 02_prompt_evaluation/              # Prompt evaluation examples
│   ├── evaluate_single_prompt.py      # Evaluate one prompt
│   ├── compare_prompts.py             # Compare multiple prompt versions
│   └── evaluate_with_judges.py        # Use LLM judges
│
├── 03_model_registry/                 # Model registration & management ✨
│   ├── register_gemini_model.py       # Register Gemini 2.5 Pro models
│   ├── load_and_predict.py            # Load and use models
│   ├── model_versioning.py            # Version & lifecycle management
│   ├── evaluate_models.py             # Comprehensive evaluation 🆕
│   ├── compare_models.py              # Side-by-side comparison 🆕
│   ├── model_ab_testing.py            # A/B testing framework 🆕
│   ├── model_performance_tracking.py  # Performance monitoring 🆕
│   ├── run_all_model_examples.sh      # Run all model examples 🆕
│   ├── README_MODEL_FEATURES.md       # Detailed documentation 🆕
│   └── MODELS_SUMMARY.md              # Implementation summary 🆕
│
├── 04_evaluation_framework/           # Evaluation scorers and framework
│   ├── custom_scorers.py              # Code-based custom scorers
│   ├── llm_judges.py                  # LLM-as-a-Judge scorers
│   ├── template_scorers.py            # Template-based scorers
│   └── versioning_scorers.py          # Version and share scorers
│
├── 05_tracing/                        # Tracing and observability
│   ├── openai_tracing.py              # OpenAI auto-tracing
│   ├── gemini_tracing.py              # Gemini tracing
│   ├── custom_tracing.py              # Custom traces
│   └── agent_tracing.py               # Agent tool-calling traces
│
├── 06_evaluation_datasets/            # Evaluation dataset management
│   ├── create_from_traces.py          # Build datasets from traces
│   ├── create_from_data.py            # Build from dictionaries
│   └── dataset_evaluation.py          # Evaluate with datasets
│
├── 07_responses_agent/                # ResponsesAgent for serving
│   ├── simple_agent.py                # Basic conversational agent
│   ├── tool_calling_agent.py          # Agent with tools
│   └── langgraph_agent.py             # Wrap LangGraph agents
│
├── 08_end_to_end/                     # Complete workflows
│   ├── full_pipeline.py               # End-to-end evaluation pipeline
│   └── production_monitoring.py       # Production monitoring patterns
│
└── utils/                             # Utility functions
    ├── __init__.py
    ├── data_utils.py                  # Data processing utilities
    └── evaluation_utils.py            # Evaluation helpers
```

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your configuration:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
```bash
# Required for model features
GEMINI_API_KEY=your_gemini_api_key

# Optional for other examples
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Start MLflow Server (Local)

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

Or simply:
```bash
mlflow ui
```

### 4. Access MLflow UI

Open your browser to: http://localhost:5000

## 🎯 Model Features (NEW!) ✨

The project now includes **comprehensive model management features** for production-ready GenAI applications:

### What's Included

1. **Model Registration**
   - Gemini 2.5 Pro models with 2M token context
   - Multiple configuration variants (balanced, creative, precise)
   - PyFunc wrapper for custom models
   - Cost-effective and powerful

2. **Model Evaluation** 🎯
   - Custom code-based scorers (4 types)
   - LLM-as-a-judge evaluators (2 types)
   - Ground truth comparison
   - Multi-metric evaluation

3. **Model Comparison** 📊
   - Side-by-side version comparison
   - Quality and performance metrics
   - Aggregate statistics

4. **A/B Testing** 🔬
   - Production-ready framework
   - User segment analysis
   - Statistical significance testing
   - Deployment recommendations

5. **Performance Monitoring** 📈
   - Time-series metrics tracking
   - Drift detection
   - Automated alerting
   - Performance reports

6. **Lifecycle Management** 🔄
   - Version control
   - Stage transitions (None → Staging → Production → Archived)
   - Tagging and search
   - Rollback support

### Run All Model Examples

```bash
cd 03_model_registry
bash run_all_model_examples.sh
```

This comprehensive demo (~10-15 minutes) showcases:
- ✅ Gemini 2.5 Pro model registration
- ✅ Comprehensive evaluation with 6 scorers
- ✅ Version comparison across diverse scenarios
- ✅ A/B testing with deployment recommendations
- ✅ 7-day performance tracking with drift detection
- ✅ Complete lifecycle management

**See:** [`03_model_registry/README_MODEL_FEATURES.md`](03_model_registry/README_MODEL_FEATURES.md) for detailed documentation.

---

## 🎯 Quick Start Examples

### Example 1: Register and Evaluate a Prompt

```python
from config import Config
import mlflow

# Setup
Config.setup_mlflow()

# Run prompt registry example
python 01_prompt_registry/register_prompts.py

# Run evaluation
python 02_prompt_evaluation/evaluate_single_prompt.py
```

### Example 2: Register and Serve a Gemini Model

```python
# Register Gemini 2.5 Pro model
python 03_model_registry/register_gemini_model.py

# Load and predict
python 03_model_registry/load_and_predict.py
```

### Example 3: Create and Use Evaluation Datasets

```python
# Create dataset from traces
python 06_evaluation_datasets/create_from_traces.py

# Evaluate using dataset
python 06_evaluation_datasets/dataset_evaluation.py
```

### Example 4: Build a ResponsesAgent

```python
# Simple conversational agent
python 07_responses_agent/simple_agent.py

# Tool-calling agent
python 07_responses_agent/tool_calling_agent.py
```

## 📊 Key MLflow Concepts

### Experiments
Top-level containers for organizing runs. Each feature demonstration uses the experiment defined in your `.env` file.

### Runs
Individual executions within an experiment. Each run logs parameters, metrics, and artifacts.

### Artifacts
Files or directories produced by runs (prompts, models, evaluation reports, datasets).

### Prompt Registry
Centralized storage for versioned prompts with lifecycle management (staging, production).

### Model Registry
Version control for models with staging and production promotion workflows.

### Traces
Detailed execution logs for LLM calls, showing inputs, outputs, latency, and token usage.

## 🔍 Feature Details

### Prompt Registry
- **Register**: `mlflow.genai.register_prompt()`
- **Load**: `mlflow.genai.load_prompt("prompts:/name@alias")`
- **Versioning**: Automatic version incrementing
- **Aliases**: latest, production, staging, etc.

### Evaluation Framework
- **Scorers**: Custom Python functions decorated with `@scorer`
- **LLM Judges**: `Guidelines()`, `Correctness()`, `make_judge()`
- **Evaluation**: `mlflow.genai.evaluate()`

### Tracing
- **Auto-trace**: `mlflow.openai.autolog()`, `mlflow.gemini.autolog()`
- **Custom traces**: `@mlflow.trace` decorator
- **Search traces**: `mlflow.search_traces()`

### Model Registry
- **Log model**: `mlflow.pyfunc.log_model()`
- **Register**: Set `registered_model_name` parameter
- **Load**: `mlflow.pyfunc.load_model()`

## 🧪 Running Tests

Each directory contains standalone examples. Run them individually:

```bash
# Test prompt registry
python 01_prompt_registry/register_prompts.py

# Test evaluation
python 02_prompt_evaluation/evaluate_single_prompt.py

# Test tracing
python 05_tracing/openai_tracing.py
```

## 📖 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow GenAI Guide](https://mlflow.org/docs/latest/llms/index.html)
- [MLflow Prompt Engineering](https://mlflow.org/docs/latest/llms/prompt-engineering/index.html)
- [MLflow Evaluation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)

## 🤝 Contributing

This is a demonstration project. Feel free to extend it with additional examples or improvements.

## 📝 License

This project is for educational and demonstration purposes.

## 🙋 Support

For issues or questions about MLflow features, please refer to the official MLflow documentation or open an issue in the MLflow GitHub repository.

---

**Built with MLflow 3.4+** | Showcasing the power of MLflow for GenAI applications

