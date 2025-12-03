# MLflow GenAI Features - Quick Start Guide

## 🚀 Get Started in 15 Minutes

This guide will walk you through setting up and running your first MLflow GenAI features demo.

---

## 📋 Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows (WSL)
- **Python**: 3.8 or later
- **Internet**: For Gemini API and MLflow server access

### What You'll Need
- Gemini API key (from Google AI Studio)
- MLflow tracking server (local or remote)
- Basic Python knowledge

---

## ⚡ 5-Minute Setup

### Step 1: Clone or Navigate to Project

```bash
cd /home/shtlp_0170/Videos/ml_flow_features
```

### Step 2: Install Dependencies

```bash
pip3 install -r requirements.txt
```

**What gets installed**:
- `mlflow` - MLOps platform
- `google-generativeai` - Gemini API client
- `pandas` - Data manipulation
- `python-dotenv` - Environment management

### Step 3: Configure Environment

Create `.env` file:

```bash
# Copy the example
cp .env.example .env

# Edit with your values
nano .env
```

**Required configuration**:

```bash
# Gemini API Key (Get from https://makersuite.google.com/app/apikey)
GEMINI_API_KEY=your_gemini_api_key_here

# MLflow Tracking (use local or remote)
MLFLOW_TRACKING_URI=http://localhost:5000
# OR for remote server:
# MLFLOW_TRACKING_URI=https://mlflow.shorthills.ai
# MLFLOW_TRACKING_USERNAME=your_username
# MLFLOW_TRACKING_PASSWORD=your_password

# Experiment Name
MLFLOW_EXPERIMENT_NAME=gemini_qa_demo
```

### Step 4: Test Configuration

```bash
python3 test_remote_connection.py
```

**Expected output**:
```
✅ MLflow configured:
   Tracking URI: http://localhost:5000
   Experiment: gemini_qa_demo

✅ Gemini API key found

✅ Connection successful!
```

---

## 🎯 Your First MLflow GenAI Workflow

### Workflow 1: Prompt Management (5 minutes)

**Goal**: Register and version prompts like code

#### Step 1: Register Prompts

```bash
cd 01_prompt_registry
python3 register_prompts.py
```

**What this does**:
1. Creates 5 different prompts in MLflow
2. Registers 2 versions of `qa_prompt_chat`
3. Sets aliases: `champion`, `baseline`, `challenger`

**Expected output**:
```
1️⃣  Registering a simple text prompt...
✅ Registered prompt: qa_prompt_simple

2️⃣  Registering a chat-style prompt...
✅ Registered chat prompt: qa_prompt_chat (Version 1)

3️⃣  Creating version 2...
✅ Created new version: 2

5️⃣  Setting aliases...
✅ Set 'champion' alias to version 2
```

#### Step 2: Load and Use Prompts

```bash
python3 simple_fetch_by_alias.py
```

**What this does**:
1. Loads prompt by alias (`champion`)
2. Displays template structure
3. Formats with example question

**Expected output**:
```
Loading prompt: qa_prompt_chat (alias: champion)
✅ Loaded prompt version: 2

Template structure:
- Role: system
  Content: You are a knowledgeable AI assistant...
- Role: user
  Content: Question: {{question}}...

✅ Formatted prompt ready for LLM
```

#### Step 3: View in MLflow UI

```bash
# If using local MLflow, start server in new terminal:
mlflow server --host 0.0.0.0 --port 5000

# Open browser
# Navigate to: http://localhost:5000
```

1. Click **"Prompts"** in sidebar
2. Find `qa_prompt_chat`
3. See versions and aliases
4. Click version to see template

**🎉 Success!** You've mastered prompt versioning!

---

### Workflow 2: Model Registration (5 minutes)

**Goal**: Register Gemini models with different configurations

#### Step 1: Register Models

```bash
cd ../03_model_registry
python3 register_gemini_model.py
```

**What this does**:
1. Creates 3 Gemini model variants:
   - **Balanced** (temp=0.7) → General purpose
   - **Creative** (temp=1.0) → Creative writing
   - **Precise** (temp=0.2) → Technical accuracy
2. Logs rich metadata (tags, metrics, params)
3. Tests each model
4. Sets stages (Production, Staging)

**Expected output**:
```
1️⃣  Registering base Gemini 2.0 Flash model...
✅ Registered model: gemini_qa_model (Version 1)

2️⃣  Registering creative Gemini variant...
✅ Registered creative variant (Version 2)

3️⃣  Registering precise Gemini variant...
✅ Registered precise variant (Version 3)

4️⃣  Testing registered models...
✅ All versions working!

5️⃣  Managing model stages...
✅ Version 1 → Production
✅ Version 2 → Staging
```

#### Step 2: Load and Predict

```bash
python3 load_and_predict.py
```

**What this does**:
1. Loads model from registry
2. Makes single prediction
3. Makes batch predictions
4. Compares versions

**Expected output**:
```
Loading model: gemini_qa_model/Production

Single prediction:
Q: What is AI?
A: Artificial intelligence refers to...

Batch prediction:
✅ Processed 3 questions
```

#### Step 3: View Models in UI

1. Navigate to MLflow UI
2. Click **"Models"** in sidebar
3. Find `gemini_qa_model`
4. See:
   - 3 versions
   - Rich tags (variant, use_case, etc.)
   - Metrics (latency, quality, etc.)
   - Current stages

**🎉 Success!** You've mastered model versioning!

---

### Workflow 3: Model Evaluation (5 minutes)

**Goal**: Compare model quality with automated metrics

#### Step 1: Evaluate Models

```bash
python3 compare_models.py
```

**What this does**:
1. Loads all 3 model versions
2. Runs on test dataset
3. Uses custom scorers
4. Compares results side-by-side

**Expected output**:
```
Evaluating version 1 (Balanced)...
✅ Quality: 0.85

Evaluating version 2 (Creative)...
✅ Quality: 0.78

Evaluating version 3 (Precise)...
✅ Quality: 0.92

📊 Comparison:
Version 1: ⭐⭐⭐⭐ (4/5)
Version 2: ⭐⭐⭐⭐ (4/5) - Most creative
Version 3: ⭐⭐⭐⭐⭐ (5/5) - Most accurate

🏆 Recommended: Version 3 for production
```

#### Step 2: View Evaluation Results

1. Go to MLflow UI
2. Click **"Experiments"**
3. Find evaluation runs
4. Compare metrics across versions

**🎉 Success!** You've mastered model evaluation!

---

## 🎓 Next Steps

Now that you've completed the basics, explore advanced features:

### 1. **Prompt Evaluation**
```bash
cd 02_prompt_evaluation
python3 compare_prompts.py
```
Learn: How to A/B test different prompts

### 2. **A/B Testing**
```bash
cd 03_model_registry
python3 model_ab_testing.py
```
Learn: Production testing framework

### 3. **LLM Tracing**
```bash
cd 05_tracing
python3 gemini_tracing.py
```
Learn: Debug and monitor LLM calls

### 4. **Custom Evaluators**
```bash
cd 04_evaluation_framework
python3 custom_scorers.py
```
Learn: Build domain-specific quality metrics

---

## 📚 Understanding the Concepts

### What is a Prompt Registry?

Think of it like **Git for prompts**:
- **Version Control**: Track changes to prompts
- **Branching**: Test variants without breaking production
- **Tagging**: Aliases point to specific versions (like Git tags)
- **History**: See what changed and why

**Real-world example**:
```python
# Development
prompt_v1 = "Be helpful."
# → Register, test, not great

prompt_v2 = "Be helpful and concise."
# → Register, test, better!

# Production
prompt = load_prompt("support_bot", alias="champion")
# → Always gets best version, no code changes needed
```

### What is a Model Registry?

Think of it like **App Store for models**:
- **Catalog**: All models in one place
- **Versioning**: Multiple versions, pick the best
- **Stages**: Development → Staging → Production
- **Metadata**: Know what each model does

**Real-world example**:
```python
# Register model variants
register_model(temperature=0.2)  # v1: Precise
register_model(temperature=0.7)  # v2: Balanced
register_model(temperature=1.0)  # v3: Creative

# Evaluate and promote
evaluate_all()
promote_to_production(version=2)  # Balanced wins

# Production code (never changes!)
model = load_model("qa_bot", stage="Production")
# → Automatically uses v2
```

### What is Model Evaluation?

Think of it like **Automated QA Testing**:
- **Unit Tests**: Custom scorers check specific things
- **User Acceptance Testing**: LLM judges rate quality
- **Regression Testing**: Compare new vs old
- **Metrics**: Quantify quality improvements

**Real-world example**:
```python
# Old way (manual)
for question in test_set:
    response = model.predict(question)
    # Human reads and rates (slow, inconsistent)

# New way (automated)
results = mlflow.evaluate(
    model=model,
    data=test_set,
    evaluators=[relevance_scorer, helpfulness_judge]
)
# → Instant, consistent, repeatable scores
```

---

## 🐛 Common Issues & Solutions

### Issue 1: API Key Not Working

**Symptom**: `401 Unauthorized` or `API key not found`

**Solution**:
```bash
# Check .env file exists
ls -la .env

# Check key is loaded
python3 -c "from config import Config; print(Config.GEMINI_API_KEY)"

# Get new key if needed
# Visit: https://makersuite.google.com/app/apikey
```

---

### Issue 2: MLflow Server Not Running

**Symptom**: `Connection refused` or `Cannot connect`

**Solution for local server**:
```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# In another terminal, run scripts
```

**Solution for remote server**:
```bash
# Check credentials
echo $MLFLOW_TRACKING_USERNAME
echo $MLFLOW_TRACKING_PASSWORD

# Test connection
python3 test_remote_connection.py
```

---

### Issue 3: Module Not Found

**Symptom**: `ModuleNotFoundError: No module named 'mlflow'`

**Solution**:
```bash
# Install requirements
pip3 install -r requirements.txt

# Verify installation
python3 -c "import mlflow; print(mlflow.__version__)"
```

---

### Issue 4: Permission Denied (Model Registry)

**Symptom**: `403 Permission denied` when registering models

**Solution**:
```bash
# Option 1: Use local MLflow (easiest)
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow server --host 0.0.0.0 --port 5000

# Option 2: Contact admin for permissions on remote server

# Option 3: Use local artifact directory
export MLFLOW_ARTIFACT_ROOT=./mlruns
```

---

### Issue 5: Prompts Not Found

**Symptom**: `Prompt 'qa_prompt_chat' not found`

**Solution**:
```bash
# Register prompts first
cd 01_prompt_registry
python3 register_prompts.py

# Verify registration
python3 -c "import mlflow; print(mlflow.genai.list_prompts())"
```

---

## 📊 Navigating the MLflow UI

### Home Page
- **Recent Runs**: Latest experiments
- **Quick Actions**: Create experiment, etc.

### Experiments Page
- **All Runs**: Every experiment run
- **Compare**: Select multiple runs to compare
- **Visualize**: Charts and graphs

### Models Page
- **Registered Models**: Your model catalog
- **Versions**: All versions of each model
- **Stages**: Current deployment status
- **Tags & Metadata**: Rich information

### Prompts Page (GenAI)
- **All Prompts**: Your prompt catalog
- **Versions**: Historical changes
- **Aliases**: Named pointers (champion, etc.)
- **Templates**: Full prompt content

### Traces Page (GenAI)
- **Recent Traces**: LLM API calls
- **Spans**: Execution breakdown
- **Inputs/Outputs**: What was sent/received
- **Timing**: Performance analysis

---

## 🎯 Best Practices for Beginners

### 1. Start Small
- Register one prompt first
- Register one model first
- Test locally before remote

### 2. Use Descriptive Names
```python
# Good
name="customer_support_friendly_v1"

# Bad
name="prompt1"
```

### 3. Always Use Aliases
```python
# Good (production code)
prompt = load_prompt("qa_prompt", alias="champion")

# Bad (hardcoded version)
prompt = load_prompt("qa_prompt", version=2)
```

### 4. Evaluate Before Deploying
```python
# Always do this
results = evaluate(model)
if results["quality"] > 0.8:
    promote_to_production()
```

### 5. Track Everything
```python
with mlflow.start_run():
    mlflow.log_param("temperature", 0.7)
    mlflow.log_metric("quality", 0.85)
    # Now you know what settings produced what results!
```

---

## 🚀 Production Deployment Pattern

Once you've tested everything, here's how to deploy:

```python
# config.py - Environment configuration
class Config:
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# app.py - Your application
from mlflow import genai, pyfunc
from config import Config

# Setup
mlflow.set_tracking_uri(Config.MLFLOW_URI)

# Load prompt (always gets latest champion version)
prompt = genai.load_prompt("qa_prompt_chat", alias="champion")

# Load model (always gets latest production version)
model = pyfunc.load_model("models:/gemini_qa_model/Production")

# Handle user request
def handle_question(user_question):
    # Format prompt
    formatted = format_template(prompt.template, question=user_question)
    
    # Generate response
    response = model.predict({"question": user_question})
    
    return response

# To update:
# 1. Register new prompt/model version
# 2. Evaluate thoroughly
# 3. Update alias/stage in MLflow UI or CLI
# 4. No code changes needed!
```

---

## 📖 Learning Path

### Week 1: Basics
- ✅ Complete this quickstart
- ✅ Explore MLflow UI
- ✅ Register prompts and models
- ✅ Make predictions

### Week 2: Evaluation
- Read `02_prompt_evaluation/README.md`
- Read `04_evaluation_framework/README.md`
- Run evaluation examples
- Create custom scorers

### Week 3: Advanced Features
- Read `05_tracing/README.md`
- Implement A/B testing
- Set up monitoring
- Build production workflows

### Week 4: Production
- Read `ARCHITECTURE.md`
- Review best practices
- Deploy to production
- Set up CI/CD integration

---

## 🆘 Getting Help

### Documentation
- **Main Docs**: `DOCUMENTATION.md` (comprehensive guide)
- **Architecture**: `ARCHITECTURE.md` (system design)
- **Feature Guides**: Each directory has `README.md`

### MLflow Resources
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [MLflow GenAI Guide](https://mlflow.org/docs/latest/llms/genai/)
- [MLflow Examples](https://github.com/mlflow/mlflow/tree/master/examples)

### Gemini Resources
- [Gemini API Docs](https://ai.google.dev/docs)
- [API Key Setup](https://makersuite.google.com/app/apikey)
- [Model Guide](https://ai.google.dev/models/gemini)

---

## 🎉 Congratulations!

You've completed the quickstart! You now know how to:
- ✅ Register and version prompts
- ✅ Register and version models
- ✅ Evaluate model quality
- ✅ Use the MLflow UI
- ✅ Deploy with stages and aliases

**Next**: Dive deeper into specific features or start building your own application!

---

## 📋 Quick Reference

### Essential Commands

```bash
# Register prompts
cd 01_prompt_registry && python3 register_prompts.py

# Register models
cd 03_model_registry && python3 register_gemini_model.py

# Evaluate models
python3 evaluate_models.py

# Compare versions
python3 compare_models.py

# Start MLflow UI (local)
mlflow server --host 0.0.0.0 --port 5000
```

### Essential Code Patterns

```python
# Load prompt by alias
prompt = mlflow.genai.load_prompt("name", alias="champion")

# Load model by stage
model = mlflow.pyfunc.load_model("models:/name/Production")

# Evaluate model
results = mlflow.evaluate(model, data, evaluators)

# Transition stage
client.transition_model_version_stage(name, version, "Production")
```

---

**Happy MLOps-ing! 🚀**


