# MLflow GenAI Features - Production-Ready Demo

A comprehensive, production-ready demonstration of MLflow GenAI features for building enterprise LLM applications with Gemini 2.0 Flash, including prompt management, model versioning, evaluation frameworks, and observability.

---

## 📚 **Documentation Hub**

### 🚀 **Getting Started**
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 15 minutes with hands-on examples
- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Comprehensive guide covering all features
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design, data flow, and technical details

### 📖 **Feature-Specific Guides**
- **[01_prompt_registry/README.md](01_prompt_registry/README.md)** - Prompt version control & management
- **[03_model_registry/README.md](03_model_registry/README.md)** - Model lifecycle & deployment

### 🎯 **Quick Navigation**
| What do you want to do? | Go here |
|--------------------------|---------|
| **Get started quickly** | [QUICKSTART.md](QUICKSTART.md) |
| **Understand the system** | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Learn all features** | [DOCUMENTATION.md](DOCUMENTATION.md) |
| **Work with prompts** | [01_prompt_registry/](01_prompt_registry/) |
| **Work with models** | [03_model_registry/](03_model_registry/) |
| **Evaluate quality** | [04_evaluation_framework/](04_evaluation_framework/) |
| **Troubleshoot issues** | [DOCUMENTATION.md#troubleshooting](DOCUMENTATION.md#troubleshooting) |

---

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
- Register Gemini 2.0 Flash-001 models using PyFunc wrapper
- 1M token context window support
- Version models with hyperparameters (3 variants: Balanced, Creative, Precise)
- Track rich metadata (tags, metrics, parameters)
- Comprehensive model evaluation (custom scorers + LLM judges)
- Side-by-side model comparison
- A/B testing framework for production
- Performance monitoring and drift detection
- Full lifecycle management (None → Staging → Production → Archived)

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
├── README.md                          # This file (project overview)
├── QUICKSTART.md                      # 15-minute getting started guide 🚀
├── DOCUMENTATION.md                   # Comprehensive documentation 📚
├── ARCHITECTURE.md                    # System design & internals 🏗️
├── config.py                          # Configuration management
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
│
├── 01_prompt_registry/                # Prompt management examples
│   ├── README.md                      # Comprehensive prompt guide 📖
│   ├── register_prompts.py            # Register and version prompts
│   ├── simple_fetch_by_alias.py       # Load prompts (production pattern)
│   ├── fetch_prompt_from_ui.py        # Advanced fetching with debugging
│   ├── advanced_prompt_usage.py       # Production workflow example
│   ├── load_prompts.py                # Basic prompt loading
│   └── prompt_lifecycle.py            # Lifecycle management
│
├── 02_prompt_evaluation/              # Prompt evaluation examples
│   ├── evaluate_single_prompt.py      # Evaluate one prompt
│   ├── compare_prompts.py             # Compare multiple prompt versions
│   └── evaluate_with_judges.py        # Use LLM judges
│
├── 03_model_registry/                 # Model registration & management ✨
│   ├── README.md                      # Comprehensive model guide 📖
│   ├── register_gemini_model.py       # Register Gemini 2.0 Flash models
│   ├── load_and_predict.py            # Load and use models
│   ├── model_versioning.py            # Version & lifecycle management
│   ├── evaluate_models.py             # Comprehensive evaluation
│   ├── compare_models.py              # Side-by-side comparison
│   └── model_ab_testing.py            # A/B testing framework
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
# Required - Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Required - MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
# OR for remote server:
# MLFLOW_TRACKING_URI=https://mlflow.shorthills.ai
# MLFLOW_TRACKING_USERNAME=your_username
# MLFLOW_TRACKING_PASSWORD=your_password

# Experiment Name
MLFLOW_EXPERIMENT_NAME=gemini_qa_demo

# Model Configuration
GEMINI_MODEL=gemini-2.0-flash-001
DEFAULT_TEMPERATURE=0.7
```

**Get Gemini API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)

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

## 🎯 Model Features ✨

The project includes **enterprise-grade model management** for production GenAI applications:

### What's Included

1. **Model Registration** (`register_gemini_model.py`)
   - Gemini 2.0 Flash-001 with 1M token context
   - 3 variants: Balanced (temp=0.7), Creative (temp=1.0), Precise (temp=0.2)
   - PyFunc wrapper for portable models
   - Rich metadata (9 metrics, 10+ tags per model)

2. **Model Evaluation** 🎯 (`evaluate_models.py`)
   - Custom code-based scorers
   - LLM-as-a-judge evaluators
   - Multi-metric evaluation
   - Ground truth comparison

3. **Model Comparison** 📊 (`compare_models.py`)
   - Side-by-side version analysis
   - Quality and performance metrics
   - Recommendation engine

4. **A/B Testing** 🔬 (`model_ab_testing.py`)
   - Production-ready framework
   - Statistical significance testing
   - Traffic splitting (50/50)
   - Deployment recommendations

5. **Performance Monitoring** 📈 (`model_performance_tracking.py`)
   - Time-series metrics tracking
   - Drift detection
   - Automated alerting
   - Performance reports

6. **Lifecycle Management** 🔄 (`model_versioning.py`)
   - Stage transitions (None → Staging → Production → Archived)
   - Version control
   - Rollback support

### Quick Start with Models

```bash
# 1. Register models (creates 3 variants)
cd 03_model_registry
python3 register_gemini_model.py

# 2. Load and predict
python3 load_and_predict.py

# 3. Evaluate and compare
python3 evaluate_models.py
python3 compare_models.py
```

**📖 Detailed Guide**: [`03_model_registry/README.md`](03_model_registry/README.md)

---

## 🎯 Quick Start Examples

### Complete Workflow (15 minutes)

**Step 1: Prompt Management** (5 min)
```bash
cd 01_prompt_registry

# Register prompts with versions and aliases
python3 register_prompts.py

# Load and use prompts
python3 simple_fetch_by_alias.py
```

**Step 2: Model Management** (5 min)
```bash
cd ../03_model_registry

# Register 3 Gemini model variants
python3 register_gemini_model.py

# Load and make predictions
python3 load_and_predict.py
```

**Step 3: Evaluation** (5 min)
```bash
# Compare model versions
python3 compare_models.py

# View results in MLflow UI
# Navigate to: http://localhost:5000
```

### Individual Feature Examples

```bash
# Prompt Registry
python3 01_prompt_registry/register_prompts.py
python3 01_prompt_registry/advanced_prompt_usage.py

# Prompt Evaluation
python3 02_prompt_evaluation/compare_prompts.py
python3 02_prompt_evaluation/evaluate_with_judges.py

# Model Evaluation
python3 03_model_registry/evaluate_models.py
python3 03_model_registry/model_ab_testing.py

# Custom Scorers
python3 04_evaluation_framework/custom_scorers.py
python3 04_evaluation_framework/llm_judges.py

# Tracing
python3 05_tracing/gemini_tracing.py
python3 05_tracing/custom_tracing.py
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

## 📖 Documentation & Resources

### 📚 Project Documentation

| Document | Description |
|----------|-------------|
| **[QUICKSTART.md](QUICKSTART.md)** | 15-minute hands-on tutorial |
| **[DOCUMENTATION.md](DOCUMENTATION.md)** | Comprehensive feature guide with use cases |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design, data flow, and internals |
| **[01_prompt_registry/README.md](01_prompt_registry/README.md)** | Prompt management deep dive |
| **[03_model_registry/README.md](03_model_registry/README.md)** | Model lifecycle management guide |

### 🔗 External Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow GenAI Guide](https://mlflow.org/docs/latest/llms/genai/index.html)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [MLflow Prompt Engineering](https://mlflow.org/docs/latest/llms/prompt-engineering/index.html)
- [MLflow Evaluation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)

### 🆘 Getting Help

1. **Quick Issues**: Check [QUICKSTART.md - Troubleshooting](QUICKSTART.md#common-issues--solutions)
2. **Technical Details**: See [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Use Cases**: See [DOCUMENTATION.md - Use Cases](DOCUMENTATION.md#use-cases)
4. **Feature Questions**: Check feature-specific READMEs in each directory

## 🤝 Contributing

This is a demonstration project. Feel free to extend it with additional examples or improvements.

## 📝 License

This project is for educational and demonstration purposes.

## 🙋 Support

For issues or questions about MLflow features, please refer to the official MLflow documentation or open an issue in the MLflow GitHub repository.

---

## 🎓 Learning Path

### Beginner (Week 1)
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Complete the 15-minute hands-on tutorial
3. Explore MLflow UI
4. Run individual examples

### Intermediate (Week 2)
1. Read [DOCUMENTATION.md](DOCUMENTATION.md)
2. Study [01_prompt_registry/README.md](01_prompt_registry/README.md)
3. Study [03_model_registry/README.md](03_model_registry/README.md)
4. Build custom evaluators

### Advanced (Week 3-4)
1. Read [ARCHITECTURE.md](ARCHITECTURE.md)
2. Implement A/B testing
3. Set up monitoring
4. Deploy to production
5. Integrate with CI/CD

---

## 🏆 Key Features

### ✅ Production-Ready
- Enterprise-grade model management
- Full lifecycle support
- Rollback capabilities
- Performance monitoring

### ✅ Well-Documented
- 3 comprehensive guides (QUICKSTART, DOCUMENTATION, ARCHITECTURE)
- Feature-specific READMEs
- Code walkthroughs
- Troubleshooting sections

### ✅ Complete Examples
- 25+ Python scripts
- 6 feature categories
- Real-world use cases
- Best practices included

### ✅ Gemini 2.0 Flash
- 1M token context window
- Fast inference (320-520ms)
- Cost-effective ($0.075/1M tokens)
- 3 pre-configured variants

---

**Built with MLflow 2.x & Gemini 2.0 Flash** | Production-ready MLOps for GenAI applications 🚀

