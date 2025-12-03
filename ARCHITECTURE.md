# System Architecture - MLflow GenAI Features

## 📐 Overview

This document explains the technical architecture, design patterns, and implementation details of the MLflow GenAI features demonstration project.

---

## 🏗️ System Components

### 1. Configuration Layer (`config.py`)

**Purpose**: Centralized configuration management

**Key Responsibilities**:
- Load environment variables from `.env`
- Validate API keys and credentials
- Setup MLflow tracking URI
- Configure authentication
- Provide default settings

**Code Flow**:
```python
# When imported
load_dotenv()  # Loads .env file
↓
Config class initialized with environment variables
↓
Config.validate()  # Checks required keys
↓
Config.setup_mlflow()  # Configures MLflow client
```

**Design Pattern**: **Singleton Pattern**
- Single source of truth for configuration
- Accessed statically via `Config.PROPERTY`
- No instantiation needed

**What Happens Internally**:
```python
import os
from dotenv import load_dotenv

# 1. Load .env file into environment
load_dotenv()  # Reads key=value pairs from .env

# 2. Read environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
# Gets value or None if not set

# 3. Validation
if not GEMINI_API_KEY:
    # Warn user about missing configuration
    
# 4. MLflow Setup
mlflow.set_tracking_uri(URI)  # Connect to server
mlflow.set_experiment(NAME)   # Create/select experiment
```

---

### 2. Prompt Registry Layer (`01_prompt_registry/`)

#### Architecture

```
┌─────────────────────────────────────────────────────┐
│              Prompt Registry Flow                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Developer                                          │
│      │                                              │
│      ├──► register_prompts.py                       │
│      │         │                                    │
│      │         ├─► mlflow.genai.register_prompt()   │
│      │         │       │                            │
│      │         │       └─► MLflow Server            │
│      │         │               │                    │
│      │         │               ├─► Store template  │
│      │         │               ├─► Assign version  │
│      │         │               └─► Create metadata │
│      │         │                                    │
│      │         └─► mlflow.genai.set_prompt_alias()  │
│      │                 │                            │
│      │                 └─► Create alias pointer     │
│      │                                              │
│      └──► simple_fetch_by_alias.py                  │
│              │                                      │
│              ├─► mlflow.genai.load_prompt()         │
│              │       │                              │
│              │       └─► Query by name + alias      │
│              │               │                      │
│              │               └─► Return template    │
│              │                                      │
│              └─► Format template with variables     │
│                      │                              │
│                      └─► Ready for LLM              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

#### Data Model

**Prompt Object**:
```python
{
    "name": "qa_prompt_chat",           # Unique identifier
    "version": 2,                       # Auto-incremented
    "template": [...],                  # Content (string or list)
    "tags": {"type": "chat"},           # Metadata
    "commit_message": "Improved tone",  # Change description
    "aliases": ["champion", "baseline"] # Named pointers
}
```

**Storage**:
- **Database**: MLflow backend database
  - Table: `prompt_registry`
  - Indexed by: (name, version)
  
- **Metadata**: JSON in database
  - Lightweight (just text)
  - Fast retrieval

#### Key Operations

**1. Registration** (`register_prompt`):
```python
# What happens internally
1. Check if prompt name exists
2. If yes: increment version
3. If no: create new entry, version=1
4. Store template as JSON
5. Save tags and commit message
6. Return prompt object
```

**2. Aliasing** (`set_prompt_alias`):
```python
# What happens internally
1. Validate alias name (not reserved)
2. Check version exists
3. Update/create alias mapping
4. Alias → Version pointer stored
```

**3. Loading** (`load_prompt`):
```python
# What happens internally
1. Query: SELECT * FROM prompts WHERE name=? AND alias=?
2. Resolve alias → version
3. Fetch template
4. Parse JSON
5. Return Prompt object with .template attribute
```

---

### 3. Prompt Evaluation Layer (`02_prompt_evaluation/`)

#### Architecture

```
┌─────────────────────────────────────────────────────┐
│           Evaluation Framework Flow                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Test Dataset (pandas DataFrame)                    │
│  ┌────────────────────────────────┐                │
│  │ question │ ground_truth │ ...  │                │
│  │ "Q1"     │ "A1"         │ ...  │                │
│  │ "Q2"     │ "A2"         │ ...  │                │
│  └────────────────────────────────┘                │
│               │                                     │
│               ▼                                     │
│  mlflow.evaluate()                                  │
│      │                                              │
│      ├─► For each row:                              │
│      │    1. Extract question                       │
│      │    2. Load prompt (if evaluating prompts)    │
│      │    3. Call model                             │
│      │    4. Capture prediction                     │
│      │    5. Run evaluators                         │
│      │    6. Log metrics                            │
│      │                                              │
│      └─► Evaluators:                                │
│           ├─► Custom Scorer                         │
│           │    ├─ Python function                   │
│           │    ├─ Receives: predictions, targets    │
│           │    └─ Returns: score                    │
│           │                                         │
│           └─► LLM Judge                             │
│                ├─ Another LLM call                  │
│                ├─ Receives: question, prediction    │
│                ├─ Guidelines for rating             │
│                └─ Returns: score + rationale        │
│                                                     │
│  Results:                                           │
│  ┌────────────────────────────────┐                │
│  │ metric_name │ value │ std      │                │
│  │ "relevance" │ 0.85  │ 0.12     │                │
│  │ "helpfulness"│ 0.78 │ 0.15     │                │
│  └────────────────────────────────┘                │
│               │                                     │
│               └─► Logged to MLflow Run              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

#### Evaluation Process

**Single Prompt Evaluation** (`evaluate_single_prompt.py`):

```python
# Step-by-step internal flow

# 1. Load prompt
prompt = mlflow.genai.load_prompt(name, alias="champion")
# → Queries MLflow server
# → Gets template

# 2. Create model function
def model_fn(input_df):
    results = []
    for question in input_df['question']:
        # Format prompt with question
        formatted = prompt.template.format(question=question)
        
        # Call LLM
        response = call_gemini(formatted)
        
        results.append(response)
    return results

# 3. Run evaluation
results = mlflow.evaluate(
    model=model_fn,        # Function to evaluate
    data=eval_data,        # Test dataset
    evaluators=[scorers]   # Quality metrics
)
# → Internally:
#   - Runs model_fn on each row
#   - Collects predictions
#   - Runs each evaluator
#   - Aggregates scores
#   - Logs to MLflow

# 4. Results stored
# → MLflow creates a run
# → Metrics saved as key-value pairs
# → Artifacts saved (predictions, etc.)
```

**Prompt Comparison** (`compare_prompts.py`):

```python
# Parallel evaluation
for prompt_alias in ["champion", "baseline", "challenger"]:
    with mlflow.start_run(run_name=f"eval_{prompt_alias}"):
        # Each gets its own run
        results = evaluate(prompt_alias)
        # Metrics logged separately

# MLflow UI shows side-by-side comparison
```

---

### 4. Model Registry Layer (`03_model_registry/`)

#### Architecture

```
┌───────────────────────────────────────────────────────┐
│              Model Registry Flow                      │
├───────────────────────────────────────────────────────┤
│                                                       │
│  1. Model Definition                                  │
│  ┌─────────────────────────────────────┐             │
│  │ class GeminiModel(PythonModel):     │             │
│  │   def __init__(self, temperature):  │             │
│  │       self.temperature = temperature│             │
│  │                                     │             │
│  │   def predict(self, ctx, input):    │             │
│  │       # Call Gemini API             │             │
│  │       response = genai.generate()   │             │
│  │       return response.text          │             │
│  └─────────────────────────────────────┘             │
│                  │                                    │
│                  ▼                                    │
│  2. Registration (log_model)                          │
│     mlflow.pyfunc.log_model(                          │
│         python_model=GeminiModel(),                   │
│         registered_model_name="gemini_qa_model"       │
│     )                                                 │
│     │                                                 │
│     ├─► Pickle model object                           │
│     ├─► Upload to artifact store                      │
│     ├─► Create registry entry                         │
│     └─► Assign version                                │
│                  │                                    │
│                  ▼                                    │
│  3. Storage                                           │
│  ┌─────────────────────────────────────┐             │
│  │ Artifact Store:                     │             │
│  │   /models/gemini_qa_model/          │             │
│  │     ├─ 1/                            │             │
│  │     │  ├─ model.pkl (pickled object)│             │
│  │     │  ├─ MLmodel (metadata)        │             │
│  │     │  └─ requirements.txt          │             │
│  │     ├─ 2/                            │             │
│  │     └─ 3/                            │             │
│  │                                     │             │
│  │ Database:                           │             │
│  │   model_registry table:             │             │
│  │     ├─ name: gemini_qa_model        │             │
│  │     ├─ version: 1, 2, 3             │             │
│  │     ├─ stage: Production/Staging    │             │
│  │     ├─ tags: {...}                  │             │
│  │     └─ metrics: {...}               │             │
│  └─────────────────────────────────────┘             │
│                  │                                    │
│                  ▼                                    │
│  4. Loading (load_model)                              │
│     model = mlflow.pyfunc.load_model(                 │
│         "models:/gemini_qa_model/Production"          │
│     )                                                 │
│     │                                                 │
│     ├─► Query database for version in Production      │
│     ├─► Download artifacts from store                 │
│     ├─► Unpickle model object                         │
│     └─► Return ready-to-use model                     │
│                  │                                    │
│                  ▼                                    │
│  5. Prediction                                        │
│     result = model.predict({"question": "..."})       │
│     │                                                 │
│     └─► Calls GeminiModel.predict()                   │
│         └─► Calls Gemini API                          │
│             └─► Returns response                      │
│                                                       │
└───────────────────────────────────────────────────────┘
```

#### PyFunc Model Wrapper

**Why PyFunc?**

1. **Standardization**: Common interface for any model
2. **Portability**: Can be moved between environments
3. **Reproducibility**: Dependencies tracked automatically
4. **Flexibility**: Wrap any Python code

**How It Works**:

```python
class GeminiModel(mlflow.pyfunc.PythonModel):
    """
    This class becomes the model artifact.
    Everything in __init__ is saved with the model.
    """
    
    def __init__(self, temperature=0.7):
        # These are saved when model is pickled
        self.temperature = temperature
        # Don't store API clients here (pickle issues)
    
    def predict(self, context, model_input):
        """
        Called when model.predict() is invoked.
        
        Args:
            context: MLflow context (has artifacts, etc.)
            model_input: User input (dict or DataFrame)
        
        Returns:
            Predictions (any Python object)
        """
        # Initialize API client here (not in __init__)
        genai.configure(api_key=Config.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash-001")
        
        # Handle input format
        if isinstance(model_input, dict):
            question = model_input["question"]
        else:  # DataFrame
            question = model_input["question"].tolist()[0]
        
        # Call API
        response = model.generate_content(
            question,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature
            )
        )
        
        return response.text

# When registered:
# 1. Python pickles the GeminiModel instance
# 2. Stores pickle file in artifact store
# 3. Creates MLmodel file with metadata
# 4. Logs to registry

# When loaded:
# 1. Downloads pickle file
# 2. Unpickles to recreate GeminiModel instance
# 3. Ready to call .predict()
```

#### Model Versioning

**Version Management**:

```python
# First registration → version 1
mlflow.pyfunc.log_model(..., registered_model_name="gemini_qa_model")
# Creates: models/gemini_qa_model/1

# Second registration → version 2
mlflow.pyfunc.log_model(..., registered_model_name="gemini_qa_model")
# Creates: models/gemini_qa_model/2

# Database tracks:
# - Version numbers (auto-incremented)
# - Creation timestamps
# - Stages (None, Staging, Production, Archived)
# - Tags and metadata
# - Source run (links to experiment run)
```

**Stage Transitions**:

```python
# Database update
UPDATE model_versions
SET current_stage = 'Production'
WHERE name = 'gemini_qa_model' AND version = 2

# Previous Production version:
UPDATE model_versions
SET current_stage = 'Archived'
WHERE name = 'gemini_qa_model' AND version = 1 AND current_stage = 'Production'

# Only one version can be in Production at a time (by default)
```

---

### 5. Evaluation Framework Layer (`04_evaluation_framework/`)

#### Custom Scorers

**Code-Based Evaluators** (`custom_scorers.py`):

```python
def relevance_scorer(predictions, targets, questions):
    """
    Pure Python function that scores predictions.
    
    Runs locally, no API calls.
    Fast but limited to programmatic checks.
    """
    scores = []
    for pred, target, question in zip(predictions, targets, questions):
        score = 0.0
        
        # Check 1: Contains key terms
        if keyword_in_response(question, pred):
            score += 0.5
        
        # Check 2: Similar to ground truth
        if similarity(pred, target) > 0.7:
            score += 0.5
        
        scores.append(score)
    
    return {"relevance": np.mean(scores)}

# Registered as evaluator:
mlflow.make_metric(
    eval_fn=relevance_scorer,
    name="custom_relevance",
    greater_is_better=True
)

# When used in evaluation:
# 1. MLflow collects all predictions
# 2. Calls relevance_scorer(predictions, targets, questions)
# 3. Gets back {"relevance": 0.85}
# 4. Logs metric to run
```

#### LLM Judges

**AI-Based Evaluators** (`llm_judges.py`):

```python
# Create judge
judge = mlflow.evaluate.make_judge(
    name="relevance_judge",
    judge_model="gemini:/gemini-1.5-flash",
    guidelines="""
    Rate relevance 1-5:
    5 = Directly answers question
    3 = Partially relevant
    1 = Off-topic
    """
)

# What happens when used:
# For each prediction:
#   1. MLflow constructs prompt:
#      f"Question: {question}
#         Answer: {prediction}
#         
#         Rate relevance using guidelines:
#         {guidelines}"
#   
#   2. Calls judge model (another Gemini)
#   
#   3. Parses response for score
#   
#   4. Optionally extracts rationale
#   
#   5. Logs score to MLflow

# Benefits:
# - Can assess subjective quality
# - Understands context and nuance
# - Flexible (just change guidelines)

# Drawbacks:
# - Slower (API call per prediction)
# - Costs money
# - Non-deterministic
```

---

### 6. Tracing Layer (`05_tracing/`)

#### Automatic Tracing

**How Tracing Works**:

```python
@mlflow.trace
def generate_answer(question):
    """
    Decorator captures function execution.
    """
    # Everything here is traced
    prompt = load_prompt("qa_prompt_chat")
    response = call_gemini(prompt.format(question=question))
    return response

# What @mlflow.trace does:
# 1. Wraps function in tracing context
# 2. Records:
#    - Function name
#    - Input arguments
#    - Start time
#    - End time
#    - Return value
#    - Exceptions (if any)
# 3. Creates span in trace
# 4. Sends to MLflow server

# Nested tracing:
@mlflow.trace
def load_prompt(name):
    # This creates a child span
    return mlflow.genai.load_prompt(name)

@mlflow.trace
def call_gemini(prompt):
    # This creates another child span
    response = genai.generate_content(prompt)
    return response

# Results in hierarchical trace:
# generate_answer [2.3s]
#  ├─ load_prompt [0.1s]
#  └─ call_gemini [2.1s]
```

**Trace Data Structure**:

```python
{
    "trace_id": "unique-uuid",
    "spans": [
        {
            "span_id": "span-1",
            "parent_span_id": null,
            "name": "generate_answer",
            "start_time": 1234567890,
            "end_time": 1234567892,
            "duration_ms": 2300,
            "inputs": {"question": "What is AI?"},
            "outputs": {"result": "AI is..."},
            "attributes": {
                "mlflow.traceRequestId": "...",
                "mlflow.traceName": "generate_answer"
            }
        },
        {
            "span_id": "span-2",
            "parent_span_id": "span-1",
            "name": "load_prompt",
            "start_time": 1234567890,
            "end_time": 1234567890.1,
            "duration_ms": 100,
            "inputs": {"name": "qa_prompt_chat"},
            "outputs": {"template": "..."}
        },
        {
            "span_id": "span-3",
            "parent_span_id": "span-1",
            "name": "call_gemini",
            "start_time": 1234567890.2,
            "end_time": 1234567892.3,
            "duration_ms": 2100,
            "inputs": {"prompt": "..."},
            "outputs": {"response": "..."},
            "attributes": {
                "llm.model": "gemini-2.0-flash-001",
                "llm.token_count.prompt": 50,
                "llm.token_count.completion": 150
            }
        }
    ]
}
```

**Storage**: Sent to MLflow tracking server, stored in database

**Visualization**: MLflow UI shows:
- Waterfall chart (timing)
- Span details (inputs/outputs)
- Errors highlighted
- Nested structure

---

## 🔄 Data Flow Diagrams

### Complete Request Flow

```
┌────────────────────────────────────────────────────────────┐
│                  Production Request Flow                   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  User Request: "What is machine learning?"                 │
│      │                                                     │
│      ▼                                                     │
│  Application Load Balancer                                 │
│      │                                                     │
│      ├─► 50% Traffic → Model Version 1 (Production)        │
│      │        │                                            │
│      │        ├─► Load model: models:/model/Production     │
│      │        │      │                                     │
│      │        │      └─► MLflow Registry                   │
│      │        │             │                              │
│      │        │             ├─► Query: Get version in Prod │
│      │        │             ├─► Response: Version 1        │
│      │        │             └─► Download artifacts         │
│      │        │                                            │
│      │        ├─► Load prompt: qa_prompt_chat/champion     │
│      │        │      │                                     │
│      │        │      └─► MLflow Prompt Registry            │
│      │        │             │                              │
│      │        │             └─► Return template            │
│      │        │                                            │
│      │        ├─► Format prompt with question              │
│      │        │                                            │
│      │        ├─► Call Gemini API                          │
│      │        │      │                                     │
│      │        │      ├─► Request (with temp=0.7)           │
│      │        │      └─► Response: "Machine learning is..."│
│      │        │                                            │
│      │        ├─► Log to MLflow (with @mlflow.trace)       │
│      │        │      │                                     │
│      │        │      ├─► Trace data (spans, timing)        │
│      │        │      ├─► Metrics (latency, cost)           │
│      │        │      └─► Tags (model_version=1)            │
│      │        │                                            │
│      │        └─► Return response to user                  │
│      │                                                     │
│      └─► 50% Traffic → Model Version 2 (Staging)           │
│               │                                            │
│               └─► [Same flow but with Version 2]           │
│                                                            │
│  Monitoring System (continuous)                            │
│      │                                                     │
│      ├─► Collect metrics from traces                       │
│      ├─► Aggregate per version                             │
│      ├─► Compare performance                               │
│      ├─► Detect anomalies                                  │
│      └─► Alert if degradation                              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 🗄️ Database Schema (Conceptual)

### MLflow Backend Database Tables

**1. `experiments`**
```sql
CREATE TABLE experiments (
    experiment_id INT PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    artifact_location VARCHAR(500),
    lifecycle_stage VARCHAR(32)
);
```

**2. `runs`**
```sql
CREATE TABLE runs (
    run_id VARCHAR(32) PRIMARY KEY,
    experiment_id INT,
    user_id VARCHAR(255),
    status VARCHAR(32),
    start_time BIGINT,
    end_time BIGINT,
    artifact_uri VARCHAR(500),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
```

**3. `params`**
```sql
CREATE TABLE params (
    run_id VARCHAR(32),
    key VARCHAR(255),
    value VARCHAR(500),
    PRIMARY KEY (run_id, key),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
```

**4. `metrics`**
```sql
CREATE TABLE metrics (
    run_id VARCHAR(32),
    key VARCHAR(255),
    value DOUBLE,
    timestamp BIGINT,
    step BIGINT DEFAULT 0,
    PRIMARY KEY (run_id, key, timestamp, step),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
```

**5. `tags`**
```sql
CREATE TABLE tags (
    run_id VARCHAR(32),
    key VARCHAR(255),
    value VARCHAR(500),
    PRIMARY KEY (run_id, key),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
```

**6. `model_versions`**
```sql
CREATE TABLE model_versions (
    name VARCHAR(255),
    version INT,
    creation_time BIGINT,
    last_updated_time BIGINT,
    description VARCHAR(5000),
    user_id VARCHAR(255),
    current_stage VARCHAR(32),
    source VARCHAR(500),
    run_id VARCHAR(32),
    status VARCHAR(32),
    PRIMARY KEY (name, version)
);
```

**7. `registered_models`**
```sql
CREATE TABLE registered_models (
    name VARCHAR(255) PRIMARY KEY,
    creation_time BIGINT,
    last_updated_time BIGINT,
    description VARCHAR(5000)
);
```

**8. `prompt_registry`** (GenAI)
```sql
CREATE TABLE prompt_registry (
    name VARCHAR(255),
    version INT,
    template TEXT,
    tags JSON,
    commit_message VARCHAR(1000),
    creation_time BIGINT,
    PRIMARY KEY (name, version)
);
```

**9. `prompt_aliases`** (GenAI)
```sql
CREATE TABLE prompt_aliases (
    name VARCHAR(255),
    alias VARCHAR(255),
    version INT,
    last_updated_time BIGINT,
    PRIMARY KEY (name, alias),
    FOREIGN KEY (name, version) REFERENCES prompt_registry(name, version)
);
```

---

## 🔐 Security Considerations

### API Key Management

**Best Practices** (Implemented):
- ✅ Store in `.env` file (not committed)
- ✅ Use environment variables
- ✅ Validate on startup
- ✅ Never log API keys

**Code Pattern**:
```python
# config.py
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# In models (not stored in pickle)
def predict(self, context, model_input):
    # Get from env at runtime
    genai.configure(api_key=Config.GEMINI_API_KEY)
```

### Authentication

**MLflow Server**:
- Basic Auth: Username/Password
- Stored in environment variables
- Transmitted over HTTPS

---

## 📊 Performance Characteristics

### Latency Breakdown

**Typical Request (Gemini 2.0 Flash)**:
```
Total: 450-520ms
├─ Load model from registry: 10-20ms (cached after first load)
├─ Load prompt: 5-10ms (cached)
├─ Format prompt: <1ms
├─ Gemini API call: 300-450ms
│  ├─ Network: 20-50ms
│  ├─ Model inference: 250-380ms
│  └─ Network return: 20-50ms
└─ Tracing overhead: 5-10ms
```

### Scalability

**MLflow Server**:
- Can handle 100s of requests/sec
- Database bottleneck for high concurrency
- Artifact store (S3/etc.) scales independently

**Gemini API**:
- Rate limits: Project-level
- Concurrent requests: Supported
- Caching: Not automatic

---

## 🛠️ Technology Stack Deep Dive

### MLflow (2.x)

**Core Components**:
1. **Tracking Server**: REST API for logging
2. **Backend Store**: Database (SQLite, PostgreSQL, MySQL)
3. **Artifact Store**: File system (local, S3, Azure, GCS)
4. **UI**: React web application
5. **Client Libraries**: Python, R, Java

**Our Usage**:
- **Tracking**: Experiments, runs, metrics
- **Registry**: Prompts and models
- **GenAI Module**: Prompt management, evaluation
- **Tracing**: Observability

### Google Gemini API

**Model: gemini-2.0-flash-001**
- **Context**: 1M tokens
- **Speed**: 300-450ms average
- **Cost**: $0.075/1M input tokens
- **Languages**: 100+ supported

**Configuration Options**:
- `temperature`: 0.0-2.0 (creativity)
- `top_p`: 0.0-1.0 (nucleus sampling)
- `top_k`: 1-40 (token sampling)
- `max_output_tokens`: Max response length

---

## 🎯 Design Patterns Used

### 1. Wrapper Pattern (PyFunc)
- Wraps external APIs in standard interface
- Enables portability and testability

### 2. Registry Pattern (Model/Prompt)
- Centralized storage and versioning
- Decouples storage from usage

### 3. Strategy Pattern (Evaluators)
- Interchangeable evaluation strategies
- Custom scorers vs LLM judges

### 4. Decorator Pattern (Tracing)
- Non-invasive instrumentation
- Transparent observability

### 5. Factory Pattern (Model Loading)
- Consistent model instantiation
- Version/stage resolution

---

## 📈 Monitoring & Observability

### Metrics Collected

**System Metrics**:
- Request latency (p50, p95, p99)
- Throughput (requests/sec)
- Error rate
- API costs

**Model Metrics**:
- Quality scores
- Token usage
- Temperature/parameters used
- Version distribution

**Business Metrics**:
- User satisfaction
- Task completion rate
- A/B test results

### Alerting Strategy

**Triggers**:
1. Latency > threshold (e.g., 1s)
2. Error rate > 5%
3. Quality score drop > 10%
4. Cost spike > 2x baseline

---

This architecture enables scalable, maintainable, and observable LLM applications! 🚀


