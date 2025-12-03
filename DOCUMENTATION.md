# MLflow GenAI Features - Comprehensive Documentation

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Core Features](#core-features)
4. [Feature-by-Feature Guide](#feature-by-feature-guide)
5. [Code Walkthrough](#code-walkthrough)
6. [Use Cases](#use-cases)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Introduction

### What is This Project?

This project demonstrates **enterprise-grade MLflow GenAI features** for managing the complete lifecycle of Large Language Model (LLM) applications. It showcases how to build production-ready AI systems with:

- **Version Control** for prompts and models
- **Evaluation Frameworks** for quality assurance
- **Tracing & Monitoring** for observability
- **A/B Testing** for optimization
- **Lifecycle Management** for deployment

### Why MLflow for GenAI?

MLflow provides a unified platform for:

| Feature | Business Value |
|---------|---------------|
| **Prompt Registry** | Version control for prompts like code |
| **Model Registry** | Centralized model management |
| **Evaluation** | Automated quality checks |
| **Tracing** | Debug and monitor LLM calls |
| **Experiment Tracking** | Compare different approaches |

### Technology Stack

- **LLM Provider**: Google Gemini 2.0 Flash-001
  - 1M token context window
  - Fast response times (320-520ms)
  - Cost-effective ($0.075 per 1M input tokens)
  
- **MLOps Platform**: MLflow 2.x
  - Prompt Registry (GenAI)
  - Model Registry
  - Experiment Tracking
  - Evaluation Framework

- **Remote Server**: `https://mlflow.shorthills.ai`
  - Centralized tracking
  - Team collaboration
  - Persistent storage

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Your Application                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐      │
│  │   Prompt     │   │    Model     │   │  Evaluation  │      │
│  │   Registry   │   │   Registry   │   │  Framework   │      │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘      │
│         │                  │                  │               │
│         └──────────────────┼──────────────────┘               │
│                            │                                   │
│                    ┌───────▼────────┐                          │
│                    │  MLflow Server │                          │
│                    │  (Tracking)    │                          │
│                    └───────┬────────┘                          │
│                            │                                   │
│         ┌──────────────────┼──────────────────┐               │
│         │                  │                  │               │
│    ┌────▼─────┐      ┌────▼─────┐      ┌────▼─────┐          │
│    │ Prompts  │      │  Models  │      │  Metrics │          │
│    │ Database │      │ Artifacts│      │   Logs   │          │
│    └──────────┘      └──────────┘      └──────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Gemini API      │
                    │  (Google AI)     │
                    └──────────────────┘
```

### Data Flow

1. **Development Phase**
   ```
   Developer → Register Prompt → MLflow Prompt Registry
   Developer → Register Model → MLflow Model Registry
   Developer → Define Evaluation → MLflow Evaluation
   ```

2. **Evaluation Phase**
   ```
   Load Prompt → Format Input → Call LLM → Capture Response
   → Run Evaluators → Log Metrics → Compare Results
   ```

3. **Production Phase**
   ```
   Load Model (by stage) → Predict → Trace Call → Monitor
   → A/B Test → Update Stage → Re-evaluate
   ```

### Component Interaction

```
┌────────────────────────────────────────────────────────────┐
│                    Logical Flow                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. Prompt Development                                     │
│     ├─ Register prompt versions                            │
│     ├─ Set aliases (champion, baseline, challenger)        │
│     └─ Load by alias or version                            │
│                                                            │
│  2. Prompt Evaluation                                      │
│     ├─ Evaluate single prompt                              │
│     ├─ Compare multiple versions                           │
│     └─ Use LLM judges for quality                          │
│                                                            │
│  3. Model Registration                                     │
│     ├─ Create PyFunc wrapper                               │
│     ├─ Register with metadata                              │
│     └─ Log parameters, tags, metrics                       │
│                                                            │
│  4. Model Evaluation                                       │
│     ├─ Custom scorers (code-based)                         │
│     ├─ LLM judges (AI-based)                               │
│     └─ Compare versions                                    │
│                                                            │
│  5. Production Deployment                                  │
│     ├─ Load model by stage                                 │
│     ├─ A/B testing framework                               │
│     ├─ Performance monitoring                              │
│     └─ Lifecycle management                                │
│                                                            │
│  6. Observability                                          │
│     ├─ Trace LLM calls                                     │
│     ├─ Monitor latency/costs                               │
│     └─ Debug issues                                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 🚀 Core Features

### 1. Prompt Registry (`01_prompt_registry/`)

**Purpose**: Version control and management for LLM prompts

**Use Case**: 
- You're building a customer service chatbot
- Need to test different prompt styles
- Want to track which prompt version performs best
- Need to rollback to previous versions if needed

**What It Does**:
- Stores prompts with versioning (like Git for prompts)
- Supports aliases (champion, baseline, challenger)
- Enables A/B testing of different prompt versions
- Tracks changes with commit messages

**Key Files**:
- `register_prompts.py` - Register new prompts
- `simple_fetch_by_alias.py` - Load prompts by alias
- `fetch_prompt_from_ui.py` - Advanced fetching with debugging
- `advanced_prompt_usage.py` - Production workflow example

---

### 2. Prompt Evaluation (`02_prompt_evaluation/`)

**Purpose**: Measure and compare prompt quality

**Use Case**:
- You have 3 different prompt versions
- Need to know which one gives better answers
- Want automated quality scoring
- Need data to make deployment decisions

**What It Does**:
- Evaluates prompts against test datasets
- Uses custom metrics (relevance, completeness, etc.)
- Uses LLM judges for subjective quality
- Compares multiple prompts side-by-side

**Key Files**:
- `evaluate_single_prompt.py` - Test one prompt
- `compare_prompts.py` - Compare multiple versions
- `evaluate_with_judges.py` - AI-powered evaluation

---

### 3. Model Registry (`03_model_registry/`)

**Purpose**: Version control and lifecycle management for LLM models

**Use Case**:
- You have Gemini models with different temperatures
- Need to test balanced vs creative vs precise variants
- Want to promote the best model to production
- Need to track model performance over time

**What It Does**:
- Registers LLM models with metadata
- Manages model stages (None → Staging → Production)
- Supports versioning with rich tags and metrics
- Enables loading models by version or stage

**Key Files**:
- `register_gemini_model.py` - Register Gemini variants
- `load_and_predict.py` - Load and use registered models
- `model_versioning.py` - Manage model lifecycle
- `compare_models.py` - Side-by-side comparison
- `evaluate_models.py` - Quality assessment
- `model_ab_testing.py` - Production A/B testing

---

### 4. Evaluation Framework (`04_evaluation_framework/`)

**Purpose**: Custom evaluation logic and LLM judges

**Use Case**:
- Built-in metrics aren't enough
- Need domain-specific quality checks
- Want to use AI to evaluate AI (LLM judges)
- Need consistent evaluation across experiments

**What It Does**:
- Defines custom scoring functions
- Creates LLM-as-a-Judge evaluators
- Provides reusable evaluation components
- Supports multi-metric evaluation

**Key Files**:
- `custom_scorers.py` - Code-based evaluators
- `llm_judges.py` - AI-powered evaluators

---

### 5. Tracing (`05_tracing/`)

**Purpose**: Monitor and debug LLM calls

**Use Case**:
- Production system is slow
- Need to see exactly what's sent to LLM
- Want to track costs per request
- Need to debug why a response is wrong

**What It Does**:
- Traces all LLM API calls
- Captures inputs, outputs, timing, costs
- Enables debugging of multi-step workflows
- Provides observability into black-box LLMs

**Key Files**:
- `gemini_tracing.py` - Trace Gemini calls
- `custom_tracing.py` - Advanced tracing patterns

---

## 📖 Feature-by-Feature Guide

### Feature 1: Prompt Registry

#### How It Works

```python
# Step 1: Register a prompt
prompt = mlflow.genai.register_prompt(
    name="qa_prompt_chat",           # Unique name
    template=chat_template,          # Prompt content
    commit_message="Initial version", # Change description
    tags={"type": "chat"}            # Metadata
)

# Step 2: Set an alias
mlflow.genai.set_prompt_alias(
    name="qa_prompt_chat",
    alias="champion",                # Named pointer
    version=prompt.version
)

# Step 3: Load by alias (production code)
loaded_prompt = mlflow.genai.load_prompt(
    name="qa_prompt_chat",
    alias="champion"                 # Always gets current champion
)
```

#### What's Happening in the Code

1. **Registration** (`register_prompts.py`):
   - Creates a new entry in MLflow's prompt database
   - Assigns a version number (auto-incremented)
   - Stores template, tags, and commit message
   - Returns a prompt object with metadata

2. **Aliases** (like Git tags):
   - `champion` - Best performing version
   - `baseline` - Stable reference version
   - `challenger` - New version being tested

3. **Loading** (`simple_fetch_by_alias.py`):
   - Queries MLflow server by name + alias
   - Retrieves the template
   - Formats with variables
   - Ready to send to LLM

#### When to Use

- **Before**: Hardcoded prompts in code files
- **After**: Centralized prompts with versions
- **Benefit**: Change prompts without code changes

---

### Feature 2: Prompt Evaluation

#### How It Works

```python
# Evaluate a prompt
results = mlflow.evaluate(
    model=lambda x: generate_response(x),  # Your LLM call
    data=eval_data,                        # Test dataset
    evaluators=[                           # Quality checks
        relevance_scorer,
        helpfulness_scorer
    ]
)
```

#### What's Happening in the Code

1. **Test Dataset** (pandas DataFrame):
   ```python
   data = {
       "question": ["What is AI?", ...],
       "ground_truth": ["AI is...", ...]
   }
   ```

2. **Evaluators** run on each output:
   - Custom scorers: Python functions
   - LLM judges: Another LLM rates the output
   - Built-in metrics: Toxicity, coherence, etc.

3. **Results**:
   - Aggregate scores (mean, median)
   - Per-example scores
   - Logged to MLflow for comparison

#### When to Use

- Testing new prompt versions
- Comparing prompt styles
- Quality assurance before production
- Finding regression issues

---

### Feature 3: Model Registry

#### How It Works

```python
# 1. Create PyFunc wrapper
class GeminiModel(mlflow.pyfunc.PythonModel):
    def __init__(self, temperature=0.7):
        self.temperature = temperature
    
    def predict(self, context, model_input):
        # Call Gemini API
        response = model.generate_content(...)
        return response.text

# 2. Register model
mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=GeminiModel(temperature=0.7),
    registered_model_name="gemini_qa_model",
    signature=signature,
    metadata={...}
)

# 3. Load in production
model = mlflow.pyfunc.load_model("models:/gemini_qa_model/Production")
result = model.predict({"question": "..."})
```

#### What's Happening in the Code

1. **PyFunc Wrapper** (`register_gemini_model.py`):
   - Wraps Gemini API in standard interface
   - Stores hyperparameters (temperature, top_p)
   - Implements `predict()` method
   - Makes model portable and reproducible

2. **Registration**:
   - Pickles the model object
   - Stores in artifact store
   - Creates registry entry
   - Logs metadata (tags, metrics, params)

3. **Loading** (`load_and_predict.py`):
   - Retrieves model from registry
   - Unpickles model object
   - Ready to make predictions
   - Can load by version or stage

#### When to Use

- Managing multiple model variants
- Promoting models through stages
- Rollback to previous versions
- Team collaboration on models

---

### Feature 4: Model Evaluation

#### How It Works

```python
# Evaluate model with custom scorers
results = mlflow.evaluate(
    model="models:/gemini_qa_model/1",
    data=eval_data,
    evaluators=[
        # Code-based scorer
        mlflow.make_metric(
            eval_fn=lambda pred, target: score,
            name="custom_metric"
        ),
        # LLM judge
        mlflow.evaluate.make_judge(
            name="relevance_judge",
            judge_model="gemini:/gemini-1.5-flash"
        )
    ]
)
```

#### What's Happening in the Code

1. **Custom Scorers** (`custom_scorers.py`):
   ```python
   def custom_scorer(predictions, targets):
       # Your logic here
       score = calculate_score(predictions, targets)
       return score
   ```
   - Runs Python code
   - Compares prediction to ground truth
   - Returns numeric score

2. **LLM Judges** (`llm_judges.py`):
   ```python
   judge = mlflow.evaluate.make_judge(...)
   ```
   - Uses another LLM to evaluate
   - Provides criteria/guidelines
   - Gets subjective quality scores
   - More flexible than code

3. **Results**:
   - Stored in MLflow runs
   - Comparable across versions
   - Visualized in UI

#### When to Use

- Comparing model versions
- Quality gates before promotion
- Regression testing
- Performance benchmarking

---

### Feature 5: Model Lifecycle Management

#### How It Works

```python
# Transition model stages
client = MlflowClient()

# Promote to Staging
client.transition_model_version_stage(
    name="gemini_qa_model",
    version=2,
    stage="Staging"
)

# Promote to Production
client.transition_model_version_stage(
    name="gemini_qa_model",
    version=2,
    stage="Production"
)

# Archive old version
client.transition_model_version_stage(
    name="gemini_qa_model",
    version=1,
    stage="Archived"
)
```

#### Stages Explained

| Stage | Purpose | Use Case |
|-------|---------|----------|
| **None** | New versions | Just registered, being tested |
| **Staging** | Pre-production | Being validated, A/B tested |
| **Production** | Live | Serving real traffic |
| **Archived** | Deprecated | No longer used, kept for records |

#### What's Happening in the Code

1. **Stage Transition** (`model_versioning.py`):
   - Updates model metadata
   - Changes stage label
   - Logs transition event
   - Notifies monitoring systems

2. **Loading by Stage**:
   ```python
   # Always gets production model
   model = mlflow.pyfunc.load_model("models:/gemini_qa_model/Production")
   ```
   - Decouples code from versions
   - Promotes without code changes
   - Enables safe rollbacks

#### When to Use

- Gradual rollout of new versions
- Blue-green deployments
- Emergency rollbacks
- Clear production/staging separation

---

### Feature 6: A/B Testing

#### How It Works

```python
# Split traffic between versions
def route_request(user_id):
    # 50/50 split
    if hash(user_id) % 2 == 0:
        model = load_model("models:/gemini_qa_model/1")
    else:
        model = load_model("models:/gemini_qa_model/2")
    
    result = model.predict(question)
    
    # Log which version was used
    mlflow.log_param("model_version", version)
    mlflow.log_metric("latency", latency)
    
    return result
```

#### What's Happening in the Code

1. **Traffic Splitting** (`model_ab_testing.py`):
   - Routes users to different versions
   - Uses consistent hashing (same user → same version)
   - Logs version assignments
   - Tracks per-version metrics

2. **Metrics Collection**:
   - Response time
   - User satisfaction
   - Error rates
   - Business metrics

3. **Statistical Analysis**:
   - Compares version performance
   - Calculates significance
   - Determines winner
   - Provides promotion recommendation

#### When to Use

- Testing new model versions
- Measuring real-world impact
- Data-driven deployment decisions
- Risk mitigation for changes

---

### Feature 7: Performance Monitoring

#### How It Works

```python
# Simulate production load
for day in range(30):
    for request in requests:
        # Make prediction
        response = model.predict(question)
        
        # Log metrics
        mlflow.log_metric("latency", latency, step=day)
        mlflow.log_metric("cost", cost, step=day)
        mlflow.log_metric("quality", quality, step=day)
    
    # Check for drift
    if detect_drift(recent_metrics):
        send_alert("Model performance degraded!")
```

#### What's Happening in the Code

1. **Metric Tracking** (`model_performance_tracking.py`):
   - Time-series data collection
   - Multiple metric types (latency, cost, quality)
   - Step-based logging (per day/hour)
   - Stored in MLflow

2. **Drift Detection**:
   - Compares recent vs baseline performance
   - Statistical tests (t-test, chi-square)
   - Threshold-based alerts
   - Automated notifications

3. **Reporting**:
   - Performance summaries
   - Trend analysis
   - Anomaly detection
   - Actionable insights

#### When to Use

- Production monitoring
- Early problem detection
- Capacity planning
- SLA compliance

---

### Feature 8: Tracing

#### How It Works

```python
@mlflow.trace
def generate_answer(question):
    # Traced automatically
    prompt = load_prompt("qa_prompt_chat")
    response = call_gemini(prompt.format(question=question))
    return response

# View trace in MLflow UI
```

#### What's Happening in the Code

1. **Automatic Tracing** (`gemini_tracing.py`):
   - Decorator captures function calls
   - Records inputs and outputs
   - Measures execution time
   - Tracks LLM API calls

2. **Nested Spans**:
   ```
   generate_answer [total: 2.3s]
   ├─ load_prompt [0.1s]
   ├─ format_prompt [0.01s]
   └─ call_gemini [2.1s]
      ├─ api_request [2.0s]
      └─ parse_response [0.1s]
   ```

3. **Trace Data**:
   - Function arguments
   - Return values
   - Exceptions
   - Timing breakdown

#### When to Use

- Debugging slow requests
- Understanding multi-step flows
- Cost attribution
- Performance optimization

---

## 💼 Use Cases

### Use Case 1: Customer Service Chatbot

**Scenario**: Building a chatbot for customer support

**Workflow**:

1. **Prompt Development**:
   ```python
   # Register different prompt styles
   - Friendly tone (champion)
   - Professional tone (baseline)
   - Empathetic tone (challenger)
   ```

2. **Evaluation**:
   ```python
   # Compare prompts on test conversations
   - Customer satisfaction score
   - Resolution rate
   - Response time
   ```

3. **Model Selection**:
   ```python
   # Test Gemini variants
   - Balanced (temp=0.7)
   - Precise (temp=0.2) for technical queries
   - Creative (temp=1.0) for open-ended questions
   ```

4. **Production**:
   ```python
   # Deploy best performers
   - Friendly prompt (champion)
   - Balanced model (Production stage)
   - Monitor customer satisfaction
   ```

5. **Monitoring**:
   ```python
   # Track metrics
   - Average response time
   - Cost per conversation
   - Customer ratings
   - Issue resolution rate
   ```

---

### Use Case 2: Content Generation Platform

**Scenario**: Automated blog post generation

**Workflow**:

1. **Prompt Engineering**:
   ```python
   # Create specialized prompts
   - SEO-optimized intro (champion)
   - Listicle format (baseline)
   - Storytelling style (challenger)
   ```

2. **Quality Evaluation**:
   ```python
   # Use LLM judges
   - Readability score
   - SEO quality
   - Originality check
   - Brand voice alignment
   ```

3. **Model Variants**:
   ```python
   # Register creative models
   - High temperature (temp=1.0)
   - Long responses (max_tokens=2048)
   - Version for each content type
   ```

4. **A/B Testing**:
   ```python
   # Split content types
   - 50% with champion prompt
   - 50% with challenger
   - Measure engagement metrics
   ```

---

### Use Case 3: Code Documentation Generator

**Scenario**: Auto-generating code documentation

**Workflow**:

1. **Prompt Design**:
   ```python
   # Technical prompts
   - API documentation style
   - Tutorial format
   - Quick reference format
   ```

2. **Model Configuration**:
   ```python
   # Precise variant
   - Low temperature (0.2)
   - Consistent formatting
   - Technical accuracy focus
   ```

3. **Evaluation**:
   ```python
   # Code-based scorers
   - Markdown validity
   - Code example accuracy
   - Completeness check
   ```

4. **Integration**:
   ```python
   # CI/CD pipeline
   - Trace generation calls
   - Monitor costs
   - Track quality trends
   ```

---

## 🎯 Best Practices

### 1. Prompt Management

**DO**:
- ✅ Use descriptive prompt names (`customer_support_friendly`)
- ✅ Write clear commit messages
- ✅ Use aliases for deployment (`champion`, `baseline`)
- ✅ Version prompts incrementally
- ✅ Tag prompts with metadata

**DON'T**:
- ❌ Hardcode prompts in application code
- ❌ Skip commit messages
- ❌ Use reserved aliases (`latest`, `production`)
- ❌ Delete old prompt versions

---

### 2. Model Registration

**DO**:
- ✅ Log comprehensive metadata (tags, metrics, params)
- ✅ Include input/output schemas
- ✅ Test models before registration
- ✅ Use semantic versioning in descriptions
- ✅ Document model variants

**DON'T**:
- ❌ Register without metadata
- ❌ Skip signature definition
- ❌ Use generic names (`model_1`, `model_2`)
- ❌ Forget to log hyperparameters

---

### 3. Evaluation

**DO**:
- ✅ Use diverse test datasets
- ✅ Combine code-based and LLM judges
- ✅ Set quality thresholds
- ✅ Evaluate before promotion
- ✅ Track evaluation metrics over time

**DON'T**:
- ❌ Use tiny test datasets
- ❌ Rely only on manual testing
- ❌ Skip regression testing
- ❌ Ignore edge cases

---

### 4. Production Deployment

**DO**:
- ✅ Use staging → production flow
- ✅ Perform A/B tests
- ✅ Monitor performance continuously
- ✅ Have rollback plans
- ✅ Document deployment decisions

**DON'T**:
- ❌ Deploy directly to production
- ❌ Skip A/B testing
- ❌ Ignore monitoring
- ❌ Forget to set alerts

---

## 🔧 Troubleshooting

### Issue 1: Can't Load Prompt by Alias

**Symptom**: `KeyError` or prompt not found

**Solutions**:
1. Check alias exists:
   ```python
   mlflow.genai.list_prompt_aliases("qa_prompt_chat")
   ```

2. Verify prompt name:
   ```python
   mlflow.genai.list_prompts()
   ```

3. Use version instead:
   ```python
   mlflow.genai.load_prompt("qa_prompt_chat", version=1)
   ```

---

### Issue 2: Model Registry Permission Denied

**Symptom**: `403 Permission denied`

**Solutions**:
1. Use local MLflow:
   ```bash
   export MLFLOW_TRACKING_URI=http://localhost:5000
   mlflow server --host 0.0.0.0
   ```

2. Contact MLflow admin for permissions

3. Check authentication:
   ```bash
   echo $MLFLOW_TRACKING_USERNAME
   echo $MLFLOW_TRACKING_PASSWORD
   ```

---

### Issue 3: Gemini API Rate Limits

**Symptom**: `429 Too Many Requests`

**Solutions**:
1. Add retry logic with exponential backoff
2. Reduce request rate
3. Use batch processing
4. Implement caching

---

### Issue 4: Evaluation Takes Too Long

**Symptom**: Evaluation runs for hours

**Solutions**:
1. Reduce test dataset size
2. Use faster models for judges (Gemini Flash)
3. Parallelize evaluations
4. Cache LLM judge responses

---

## 📚 Additional Resources

### Documentation Files

- `README.md` - Project overview and quick start
- `ARCHITECTURE.md` - Detailed system architecture
- `FEATURES_GUIDE.md` - Feature-specific deep dives
- `QUICKSTART.md` - Step-by-step getting started
- Directory READMEs - Module-specific guides

### External Links

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Gemini API Docs](https://ai.google.dev/docs)
- [MLflow GenAI](https://mlflow.org/docs/latest/llms/genai/index.html)

---

## 🎓 Next Steps

1. **Read** `QUICKSTART.md` for hands-on tutorial
2. **Run** examples in order (01 → 08)
3. **Explore** MLflow UI at your tracking URI
4. **Customize** for your use case
5. **Deploy** to production

---

**Need Help?** Check the troubleshooting section or reach out to your team!


