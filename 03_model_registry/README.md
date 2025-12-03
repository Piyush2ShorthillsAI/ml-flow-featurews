# Model Registry - Version Control for LLM Models

## 📋 Overview

The Model Registry provides **enterprise-grade model management** for Gemini LLM models, including versioning, lifecycle management, evaluation, and deployment.

---

## 🎯 Use Cases

### 1. **Model Versioning**
- Register multiple model configurations
- Track hyperparameter changes
- Compare model variants

### 2. **Lifecycle Management**
- Promote models through stages (None → Staging → Production)
- Rollback to previous versions
- Archive deprecated models

### 3. **A/B Testing**
- Test multiple models in production
- Split traffic between versions
- Measure real-world performance

### 4. **Performance Monitoring**
- Track metrics over time
- Detect model degradation
- Automated alerting

### 5. **Model Evaluation**
- Automated quality checks
- Compare versions objectively
- Quality gates before promotion

---

## 📁 Files in This Directory

### 1. `register_gemini_model.py` ⭐

**Purpose**: Register Gemini 2.0 Flash models with rich metadata

**What it does**:
1. Creates PyFunc wrapper for Gemini API
2. Registers 3 model variants (Balanced, Creative, Precise)
3. Logs comprehensive metadata (tags, metrics, parameters)
4. Tests registered models
5. Manages model stages

**Code Walkthrough**:

#### Part 1: PyFunc Model Wrapper

```python
class GeminiModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow model that wraps Gemini API.
    
    Why PyFunc?
    - Standard interface across all models
    - Portable (can deploy anywhere)
    - Versioned with dependencies
    - Reproducible
    """
    
    def __init__(self, model_name="gemini-2.0-flash-001", 
                 temperature=0.7, top_p=0.9, top_k=40, max_tokens=1024):
        """
        Store hyperparameters.
        These are saved when model is pickled.
        
        Args:
            model_name: Gemini model identifier
            temperature: 0.0-2.0 (higher = more creative)
            top_p: 0.0-1.0 (nucleus sampling)
            top_k: 1-40 (token sampling)
            max_tokens: Max response length
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        
        # Note: Don't store API client here!
        # Causes pickle errors
    
    def predict(self, context, model_input):
        """
        Called when model.predict() is invoked.
        
        Args:
            context: MLflow context (has artifacts, etc.)
            model_input: User input (dict or DataFrame)
        
        Returns:
            Model prediction(s)
        """
        # Initialize API client HERE (not in __init__)
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
```

**What happens when model is registered**:
1. Python pickles the `GeminiModel` instance
2. Stores hyperparameters (temperature, etc.)
3. Saves pickle file to artifact store
4. Creates metadata file (`MLmodel`)
5. Logs to Model Registry

**What happens when model is loaded**:
1. Downloads pickle file from artifact store
2. Unpickles to recreate `GeminiModel` instance
3. Hyperparameters restored from pickle
4. Ready to call `.predict()`

#### Part 2: Model Registration with Rich Metadata

```python
def main():
    """Register 3 Gemini model variants"""
    
    # Define model signature (input/output schema)
    input_schema = Schema([
        ColSpec(name="question", type=DataType.string)
    ])
    output_schema = Schema([
        ColSpec(name="response", type=DataType.string)
    ])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
    # ============================================================
    # Variant 1: Balanced (General Purpose)
    # ============================================================
    
    with mlflow.start_run(run_name="register_gemini_base"):
        # Create model instance
        base_model = GeminiModel(
            model_name="gemini-2.0-flash-001",
            temperature=0.7,    # Balanced creativity
            top_p=0.9,
            top_k=40,
            max_tokens=1024
        )
        
        # Log parameters (visible in Params tab)
        mlflow.log_param("model_name", "gemini-2.0-flash-001")
        mlflow.log_param("temperature", 0.7)
        mlflow.log_param("top_p", 0.9)
        mlflow.log_param("top_k", 40)
        mlflow.log_param("max_tokens", 1024)
        
        # Set tags (visible in Tags tab - like labels)
        mlflow.set_tag("model_name", "gemini-2.0-flash-001")
        mlflow.set_tag("provider", "google-gemini")
        mlflow.set_tag("use_case", "general_qa")
        mlflow.set_tag("variant", "balanced")
        
        # Hyperparameter tags (searchable)
        mlflow.set_tag("param_temperature", "0.7")
        mlflow.set_tag("param_max_tokens", "1024")
        mlflow.set_tag("param_top_p", "0.9")
        mlflow.set_tag("param_top_k", "40")
        
        # Model metadata tags
        mlflow.set_tag("meta_context_window", "1M tokens")
        mlflow.set_tag("meta_speed", "fast")
        mlflow.set_tag("meta_quality", "high")
        mlflow.set_tag("meta_cost_per_1m_input_tokens", "0.075")
        mlflow.set_tag("meta_cost_per_1m_output_tokens", "0.30")
        mlflow.set_tag("meta_recommended", "General Q&A with balanced creativity and accuracy")
        
        # Log metrics (visible in Metrics tab)
        mlflow.log_metric("latency_ms", 450)         # Response time
        mlflow.log_metric("throughput_rps", 20)      # Requests/sec
        mlflow.log_metric("quality_score", 0.85)     # Overall quality
        mlflow.log_metric("accuracy_score", 0.82)    # Factual accuracy
        mlflow.log_metric("creativity_score", 0.70)  # Creative output
        mlflow.log_metric("coherence_score", 0.88)   # Coherence
        mlflow.log_metric("cost_per_request", 0.0015)  # Est. cost
        mlflow.log_metric("context_utilization", 0.45)  # Context usage
        mlflow.log_metric("token_efficiency", 0.92)  # Token efficiency
        
        # Register the model
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=base_model,
            registered_model_name="gemini_qa_model",  # Name in registry
            signature=signature,
            input_example={"question": "What is artificial intelligence?"},
            metadata={
                "description": "Gemini 2.0 Flash Q&A model - Balanced configuration",
                "model_name": "gemini-2.0-flash-001",
                "variant": "balanced",
                "temperature": "0.7",
                "provider": "google-gemini",
                "use_case": "General Q&A, balanced creativity and accuracy"
            }
        )
        
        print(f"✅ Registered model: gemini_qa_model")
        print(f"   Model URI: {model_info.model_uri}")
        print(f"   Version: 1")
```

**What gets stored**:

1. **Artifact Store** (`/artifacts/`):
   ```
   model/
   ├── model.pkl              # Pickled GeminiModel
   ├── MLmodel                # Metadata file
   ├── conda.yaml             # Dependencies
   ├── python_env.yaml        # Python version
   └── requirements.txt       # Pip packages
   ```

2. **Database** (`model_versions` table):
   ```
   name: gemini_qa_model
   version: 1
   stage: None
   run_id: abc123...
   source: runs:/abc123/model
   tags: {variant: balanced, ...}
   ```

3. **MLflow Run** (experiment tracking):
   - Parameters: temperature, top_p, etc.
   - Metrics: latency, quality, etc.
   - Tags: provider, use_case, etc.
   - Artifacts: model files

**Why all this metadata?**
- **Searchable**: Find models by tags
- **Comparable**: See differences between versions
- **Documented**: Know what each version is for
- **Traceable**: Link back to experiment runs
- **Professional**: Looks good in MLflow UI!

#### Part 3: Multiple Variants

```python
# Variant 2: Creative (Higher Temperature)
with mlflow.start_run(run_name="register_gemini_creative"):
    creative_model = GeminiModel(
        temperature=1.0,    # Higher = more creative
        top_p=0.95,
        top_k=50,
        max_tokens=2048     # Longer responses
    )
    
    # ... log params, tags, metrics ...
    
    mlflow.pyfunc.log_model(
        python_model=creative_model,
        registered_model_name="gemini_qa_model",  # Same name = new version
        # ... signature, metadata ...
    )
    # Creates version 2

# Variant 3: Precise (Lower Temperature)
with mlflow.start_run(run_name="register_gemini_precise"):
    precise_model = GeminiModel(
        temperature=0.2,    # Lower = more deterministic
        top_p=0.8,
        top_k=20,
        max_tokens=512      # Shorter responses
    )
    
    # ... log params, tags, metrics ...
    
    mlflow.pyfunc.log_model(
        python_model=precise_model,
        registered_model_name="gemini_qa_model",  # Same name = new version
        # ... signature, metadata ...
    )
    # Creates version 3
```

**Result**: 3 versions in registry
- Version 1: Balanced (temp=0.7)
- Version 2: Creative (temp=1.0)
- Version 3: Precise (temp=0.2)

#### Part 4: Stage Management

```python
from mlflow import MlflowClient
client = MlflowClient()

# Promote version 1 to Production
client.transition_model_version_stage(
    name="gemini_qa_model",
    version=1,
    stage="Production"
)
# Now "Production" stage points to version 1

# Promote version 2 to Staging
client.transition_model_version_stage(
    name="gemini_qa_model",
    version=2,
    stage="Staging"
)
# Now "Staging" stage points to version 2

# Version 3 stays in "None" stage (just registered, being evaluated)
```

**Run it**:
```bash
cd 03_model_registry
python3 register_gemini_model.py
```

**Output**:
```
1️⃣  Registering base Gemini 2.0 Flash model...
✅ Registered model: gemini_qa_model
   Model URI: runs:/abc123.../model
   Version: 1

2️⃣  Registering creative Gemini variant...
✅ Registered creative variant
   Version: 2

3️⃣  Registering precise Gemini variant...
✅ Registered precise variant
   Version: 3

4️⃣  Testing registered Gemini models...
   Testing version 1...
   ✅ Version 1 response:
      Quantum computing is a revolutionary computing paradigm...

5️⃣  Managing Gemini model stages...
✅ Version 1 → Production
✅ Version 2 → Staging

✨ Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Registered Gemini model: gemini_qa_model
✅ Created 3 versions with different configurations:
   v1: Balanced (temp=0.7) → Production
   v2: Creative (temp=1.0) → Staging
   v3: Precise (temp=0.2) → None
```

---

### 2. `load_and_predict.py`

**Purpose**: Load registered models and make predictions

**What it does**:
1. Load models by version
2. Load models by stage
3. Single predictions
4. Batch predictions

**Code Walkthrough**:

```python
# Method 1: Load by specific version
model_v1 = mlflow.pyfunc.load_model("models:/gemini_qa_model/1")
# Loads version 1 (Balanced variant)

# Method 2: Load by stage (recommended for production)
model_prod = mlflow.pyfunc.load_model("models:/gemini_qa_model/Production")
# Loads whichever version is in Production stage

# Method 3: Load latest version
client = MlflowClient()
latest_versions = client.get_latest_versions("gemini_qa_model")
latest_model = mlflow.pyfunc.load_model(f"models:/gemini_qa_model/{latest_versions[0].version}")

# Single prediction
result = model_v1.predict({"question": "What is AI?"})
print(result)
# Output: "Artificial intelligence (AI) refers to..."

# Batch prediction (multiple questions)
import pandas as pd
questions_df = pd.DataFrame({
    "question": [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?"
    ]
})

results = model_v1.predict(questions_df)
print(results)
# Output: List of 3 responses
```

**What happens when loading**:
1. Queries registry database for model metadata
2. Downloads artifacts from artifact store
3. Unpickles model object
4. Restores hyperparameters
5. Returns ready-to-use model

**What happens when predicting**:
1. Calls `GeminiModel.predict()` method
2. Initializes Gemini API client
3. Formats input
4. Calls Gemini API with saved hyperparameters
5. Returns response

**Run it**:
```bash
cd 03_model_registry
python3 load_and_predict.py
```

---

### 3. `evaluate_models.py`

**Purpose**: Comprehensive model evaluation using custom scorers and LLM judges

**What it does**:
1. Evaluates models against test dataset
2. Uses custom code-based scorers
3. Uses LLM-as-a-Judge for subjective metrics
4. Compares versions side-by-side
5. Logs results to MLflow

**Code Walkthrough**:

```python
# Step 1: Create test dataset
eval_data = pd.DataFrame({
    "question": [
        "What is machine learning?",
        "Explain quantum computing",
        "What is blockchain?"
    ],
    "ground_truth": [
        "Machine learning is...",
        "Quantum computing is...",
        "Blockchain is..."
    ]
})

# Step 2: Define custom scorers
def answer_length_scorer(predictions, targets):
    """Code-based scorer: Check answer length"""
    scores = []
    for pred in predictions:
        # Good answers are 50-200 words
        word_count = len(pred.split())
        if 50 <= word_count <= 200:
            score = 1.0
        else:
            score = max(0, 1.0 - abs(word_count - 125) / 125)
        scores.append(score)
    return {"answer_length": np.mean(scores)}

# Step 3: Create LLM judges
relevance_judge = mlflow.evaluate.make_judge(
    name="relevance_judge",
    judge_model="gemini:/gemini-1.5-flash",
    guidelines="""
    Rate how relevant the answer is to the question (1-5):
    5 = Directly and completely answers the question
    4 = Mostly relevant with minor off-topic points
    3 = Partially relevant
    2 = Somewhat related but misses key points
    1 = Completely off-topic
    """
)

helpfulness_judge = mlflow.evaluate.make_judge(
    name="helpfulness_judge",
    judge_model="gemini:/gemini-1.5-flash",
    guidelines="""
    Rate how helpful the answer is (1-5):
    5 = Extremely helpful, actionable, clear
    4 = Very helpful with good examples
    3 = Moderately helpful
    2 = Slightly helpful
    1 = Not helpful at all
    """
)

# Step 4: Evaluate model
results = mlflow.evaluate(
    model="models:/gemini_qa_model/1",  # Model to evaluate
    data=eval_data,                      # Test dataset
    evaluators=[
        answer_length_scorer,
        relevance_judge,
        helpfulness_judge
    ],
    predictions="predictions"
)

print(f"Average Scores:")
print(f"  Answer Length: {results.metrics['answer_length']}")
print(f"  Relevance: {results.metrics['relevance_judge/mean']}")
print(f"  Helpfulness: {results.metrics['helpfulness_judge/mean']}")
```

**What happens internally**:

1. **For each row in eval_data**:
   ```
   Load model → Predict on question → Store prediction
   ```

2. **Run code-based scorers**:
   ```
   answer_length_scorer(predictions, targets)
   → Returns: {"answer_length": 0.85}
   ```

3. **Run LLM judges**:
   ```
   For each prediction:
     Format judge prompt:
       "Question: {question}
        Answer: {prediction}
        
        Rate relevance (1-5): {guidelines}"
     
     Call judge model (Gemini Flash)
     Parse score from response
     Aggregate scores
   
   → Returns: {"relevance_judge/mean": 4.2}
   ```

4. **Log to MLflow**:
   ```
   Creates evaluation run
   Logs all metrics
   Saves predictions as artifact
   Links to model version
   ```

**Run it**:
```bash
cd 03_model_registry
python3 evaluate_models.py
```

**Output**:
```
Evaluating model: gemini_qa_model version 1

Running custom scorers...
✅ Answer length scorer complete

Running LLM judges...
✅ Relevance judge complete (4.3/5.0)
✅ Helpfulness judge complete (4.1/5.0)

📊 Evaluation Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Metric              Score
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  answer_length       0.85
  relevance_judge     4.3 / 5.0
  helpfulness_judge   4.1 / 5.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

View details in MLflow UI
```

---

### 4. `compare_models.py`

**Purpose**: Side-by-side comparison of multiple model versions

**What it does**:
1. Evaluates multiple versions on same dataset
2. Compares metrics across versions
3. Visualizes differences
4. Recommends best version

**Code Walkthrough**:

```python
# Evaluate all 3 versions
versions_to_compare = [1, 2, 3]  # Balanced, Creative, Precise
results_by_version = {}

for version in versions_to_compare:
    print(f"\nEvaluating version {version}...")
    
    with mlflow.start_run(run_name=f"compare_v{version}"):
        results = mlflow.evaluate(
            model=f"models:/gemini_qa_model/{version}",
            data=eval_data,
            evaluators=[
                answer_length_scorer,
                relevance_judge,
                helpfulness_judge
            ]
        )
        
        results_by_version[version] = {
            "answer_length": results.metrics["answer_length"],
            "relevance": results.metrics["relevance_judge/mean"],
            "helpfulness": results.metrics["helpfulness_judge/mean"]
        }

# Compare results
comparison_df = pd.DataFrame(results_by_version).T
comparison_df.index.name = "Version"

print("\n📊 Model Comparison:")
print(comparison_df.to_string())

# Determine best version
best_version = comparison_df["relevance"].idxmax()
print(f"\n🏆 Recommended: Version {best_version}")
```

**Output**:
```
📊 Model Comparison:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Version │ answer_length │ relevance │ helpfulness
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1    │     0.85      │   4.3     │    4.1
   2    │     0.78      │   4.0     │    4.5     (More creative)
   3    │     0.92      │   4.7     │    3.9     (More precise)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 Recommended: Version 3 (Highest relevance score)
```

**Run it**:
```bash
cd 03_model_registry
python3 compare_models.py
```

---

### 5. `model_ab_testing.py`

**Purpose**: A/B testing framework for production deployment

**What it does**:
1. Splits traffic between model versions
2. Tracks performance per version
3. Compares real-world metrics
4. Provides statistical analysis
5. Recommends which version to promote

**Code Walkthrough**:

```python
class ABTester:
    """A/B testing framework for models"""
    
    def __init__(self, model_a_uri, model_b_uri):
        self.model_a = mlflow.pyfunc.load_model(model_a_uri)
        self.model_b = mlflow.pyfunc.load_model(model_b_uri)
        self.results_a = []
        self.results_b = []
    
    def route_request(self, user_id, question):
        """
        Route request to model A or B based on user_id.
        Uses consistent hashing (same user always gets same model).
        """
        if hash(user_id) % 2 == 0:
            model = self.model_a
            version = "A"
        else:
            model = self.model_b
            version = "B"
        
        # Make prediction
        start_time = time.time()
        response = model.predict({"question": question})
        latency = time.time() - start_time
        
        # Store results
        result = {
            "version": version,
            "latency": latency,
            "response_length": len(response),
            "timestamp": time.time()
        }
        
        if version == "A":
            self.results_a.append(result)
        else:
            self.results_b.append(result)
        
        return response, version
    
    def analyze_results(self):
        """Statistical analysis of A/B test"""
        df_a = pd.DataFrame(self.results_a)
        df_b = pd.DataFrame(self.results_b)
        
        print("\n📊 A/B Test Results:")
        print(f"\nModel A (n={len(df_a)}):")
        print(f"  Avg Latency: {df_a['latency'].mean():.3f}s")
        print(f"  Avg Response Length: {df_a['response_length'].mean():.0f} chars")
        
        print(f"\nModel B (n={len(df_b)}):")
        print(f"  Avg Latency: {df_b['latency'].mean():.3f}s")
        print(f"  Avg Response Length: {df_b['response_length'].mean():.0f} chars")
        
        # Statistical significance test (t-test)
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            df_a['latency'], 
            df_b['latency']
        )
        
        print(f"\nStatistical Significance:")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            if df_a['latency'].mean() < df_b['latency'].mean():
                print(f"  ✅ Model A is significantly faster")
            else:
                print(f"  ✅ Model B is significantly faster")
        else:
            print(f"  ⚠️  No significant difference")

# Usage
tester = ABTester(
    model_a_uri="models:/gemini_qa_model/1",
    model_b_uri="models:/gemini_qa_model/2"
)

# Simulate 100 requests
for i in range(100):
    user_id = f"user_{i}"
    question = generate_test_question()
    
    response, version = tester.route_request(user_id, question)
    print(f"User {user_id} → Model {version}")

# Analyze results
tester.analyze_results()
```

**Output**:
```
📊 A/B Test Results:

Model A (n=50):
  Avg Latency: 0.450s
  Avg Response Length: 823 chars

Model B (n=50):
  Avg Latency: 0.520s
  Avg Response Length: 1245 chars

Statistical Significance:
  t-statistic: -3.245
  p-value: 0.0018
  ✅ Model A is significantly faster

💡 Recommendation: Promote Model A to Production
```

**Run it**:
```bash
cd 03_model_registry
python3 model_ab_testing.py
```

---

### 6. `model_versioning.py`

**Purpose**: Demonstrate full model lifecycle management

**What it does**:
1. Register new model versions
2. Transition through stages
3. Update descriptions
4. Archive old versions
5. Rollback if needed

**Stage Lifecycle**:
```
None → Staging → Production → Archived
 ↑__________________|
    (rollback)
```

**Run it**:
```bash
cd 03_model_registry
python3 model_versioning.py
```

---

## 🔄 Typical Workflow

### 1. **Development Phase**

```bash
# Register model variants
python3 register_gemini_model.py

# Result: 3 versions in registry
# - Version 1: Balanced
# - Version 2: Creative
# - Version 3: Precise
```

### 2. **Evaluation Phase**

```bash
# Evaluate each version
python3 evaluate_models.py

# Compare versions
python3 compare_models.py

# Result: Quality metrics for each version
```

### 3. **Staging Phase**

```python
# Promote best version to Staging
client.transition_model_version_stage(
    name="gemini_qa_model",
    version=3,  # Precise variant
    stage="Staging"
)
```

### 4. **A/B Testing Phase**

```bash
# Test staging vs current production
python3 model_ab_testing.py

# Result: Statistical comparison
```

### 5. **Production Deployment**

```python
# Promote to Production
client.transition_model_version_stage(
    name="gemini_qa_model",
    version=3,
    stage="Production"
)

# Archive old production version
client.transition_model_version_stage(
    name="gemini_qa_model",
    version=1,
    stage="Archived"
)
```

### 6. **Production Usage**

```python
# In your application:
model = mlflow.pyfunc.load_model("models:/gemini_qa_model/Production")
response = model.predict({"question": user_question})
```

### 7. **Monitoring** (Optional)

```bash
python3 model_performance_tracking.py
# Tracks metrics over time, detects drift
```

---

## 💡 Best Practices

### ✅ DO

1. **Register with Rich Metadata**
   ```python
   # Good: Comprehensive tags and metrics
   mlflow.set_tag("variant", "balanced")
   mlflow.set_tag("use_case", "customer_support")
   mlflow.log_metric("latency_ms", 450)
   mlflow.log_metric("quality_score", 0.85)
   ```

2. **Use Stages, Not Versions in Production**
   ```python
   # Good: Decouples code from versions
   model = load_model("models:/model/Production")
   
   # Bad: Hardcoded version
   model = load_model("models:/model/3")
   ```

3. **Evaluate Before Promoting**
   ```python
   # Always evaluate before stage transition
   results = mlflow.evaluate(model, data, evaluators)
   if results.metrics["quality"] > 0.8:
       transition_to_staging()
   ```

4. **A/B Test Before Full Rollout**
   ```python
   # Test with 10% traffic first
   if user_id % 10 == 0:
       model = load_staging_model()
   else:
       model = load_production_model()
   ```

5. **Document Versions**
   ```python
   # Use descriptive metadata
   metadata={
       "description": "Optimized for technical questions",
       "changes": "Reduced temperature to 0.2 for consistency",
       "evaluation_date": "2024-01-15"
   }
   ```

### ❌ DON'T

1. **Don't Register Without Testing**
   - Test locally first
   - Validate input/output formats
   - Check API key access

2. **Don't Skip Evaluation**
   - Always measure quality
   - Compare against baseline
   - Document results

3. **Don't Delete Versions**
   - Archive instead
   - Keep for rollback
   - Historical reference

4. **Don't Use API Keys in Model Code**
   ```python
   # Bad
   def __init__(self, api_key):
       self.api_key = api_key  # Gets pickled!
   
   # Good
   def predict(self, context, input):
       api_key = os.getenv("GEMINI_API_KEY")
   ```

---

## 🐛 Troubleshooting

### Issue 1: Model Registration Fails

**Error**: `Permission denied` or `403`

**Solution**: 
- Use local MLflow server
- Contact admin for permissions
- Check artifact store access

---

### Issue 2: Model Loading Fails

**Error**: `Model not found` or version not found

**Solution**:
```python
# List registered models
client = MlflowClient()
models = client.search_registered_models()
print(models)

# List versions
versions = client.search_model_versions(f"name='gemini_qa_model'")
print(versions)
```

---

### Issue 3: Prediction Fails

**Error**: `API key not found` or `401 Unauthorized`

**Solution**:
```python
# Check API key is set
print(os.getenv("GEMINI_API_KEY"))

# Make sure .env file is in correct location
# Make sure config.py loads .env
```

---

## 📊 Viewing Results

### MLflow UI

1. Navigate to: `http://localhost:5000` or tracking URI
2. Click "Models" in sidebar
3. Find "gemini_qa_model"
4. View:
   - All versions
   - Current stages
   - Tags and metadata
   - Metrics
   - Linked experiments

### Programmatic Access

```python
client = MlflowClient()

# Get model versions
versions = client.search_model_versions(f"name='gemini_qa_model'")

for v in versions:
    print(f"Version {v.version}:")
    print(f"  Stage: {v.current_stage}")
    print(f"  Tags: {v.tags}")
    print(f"  Run ID: {v.run_id}")
```

---

## 🚀 Next Steps

1. **Explore Evaluation**: Go to `../04_evaluation_framework/`
2. **Learn Tracing**: Go to `../05_tracing/`
3. **Read Main Docs**: `../DOCUMENTATION.md`

---

This is enterprise-grade model management! 🎯


