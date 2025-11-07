# Using Remote MLflow Server (https://mlflow.shorthills.ai)

## 📡 Overview

When using the remote MLflow server at `https://mlflow.shorthills.ai`, **ALL your results are stored on that server**, not locally. This is great for:

✅ **Team Collaboration** - Everyone sees the same experiments  
✅ **Centralized Storage** - All data in one place  
✅ **No Local Setup** - No need to run `mlflow ui` locally  
✅ **Persistent Results** - Data survives local machine restarts  
✅ **Shared Access** - View results from any computer  

---

## 🔐 Configuration Setup

### Step 1: Get Your Credentials

You need:
1. **Username** - Your MLflow server username
2. **Password** - Your MLflow server password
3. **Server URL** - `https://mlflow.shorthills.ai`

### Step 2: Update .env File

```bash
cd /home/shtlp_0170/Videos/ml_flow_features
cp .env.example .env
nano .env  # or your preferred editor
```

Set these values in `.env`:

```bash
# MLflow Remote Server (Required)
MLFLOW_TRACKING_URI=https://mlflow.shorthills.ai
MLFLOW_EXPERIMENT_NAME=your_name_mlflow_demo  # Use unique name!

# Authentication (Required for remote server)
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password

# LLM Configuration
GEMINI_API_KEY=your_gemini_api_key
DEFAULT_LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-pro
```

### Step 3: Important Naming Conventions

⚠️ **IMPORTANT**: Use unique names to avoid conflicts with other users!

```bash
# ❌ BAD - Generic names (may conflict)
MLFLOW_EXPERIMENT_NAME=mlflow_features_demo

# ✅ GOOD - Include your name/team
MLFLOW_EXPERIMENT_NAME=john_doe_mlflow_demo
MLFLOW_EXPERIMENT_NAME=team_alpha_genai_project
```

When registering prompts/models, also use unique names:
```python
# Include your name/identifier
prompt = mlflow.genai.register_prompt(
    name="john_doe_qa_prompt",  # Not just "qa_prompt"
    template="..."
)

model = mlflow.pyfunc.log_model(
    registered_model_name="john_doe_genai_model"  # Not just "genai_model"
)
```

---

## 🚀 Quick Start with Remote Server

### 1. Configure Environment

```bash
# Edit .env with your credentials
nano .env
```

Add:
```bash
MLFLOW_TRACKING_URI=https://mlflow.shorthills.ai
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password
MLFLOW_EXPERIMENT_NAME=yourname_demo
GEMINI_API_KEY=your_gemini_key
```

### 2. Test Connection

```bash
python -c "
from config import Config
Config.setup_mlflow()
print('✅ Connected to remote MLflow server!')
"
```

### 3. Run Examples

```bash
# No need to start local MLflow server!
# Just run the examples directly:

python quickstart_gemini.py
```

### 4. View Results

Open in browser: **https://mlflow.shorthills.ai**

Login with your credentials to see your results!

---

## 📊 Where Results Are Stored

### With Remote Server:

```
https://mlflow.shorthills.ai
├── Experiments (your_name_*)
│   ├── Runs
│   │   ├── Parameters
│   │   ├── Metrics
│   │   ├── Artifacts
│   │   └── Traces
│   └── Evaluation Results
├── Prompts (your_name_*)
│   └── Versions
└── Models (your_name_*)
    └── Versions
```

**Everything is stored on the remote server**, including:
- ✅ Experiments and runs
- ✅ Parameters and metrics
- ✅ Artifacts (files, models)
- ✅ Traces (LLM calls)
- ✅ Registered prompts
- ✅ Registered models
- ✅ Evaluation results

---

## 🔄 Local vs Remote Comparison

| Feature | Local (`http://localhost:5000`) | Remote (`https://mlflow.shorthills.ai`) |
|---------|--------------------------------|----------------------------------------|
| **Setup** | Run `mlflow ui --port 5000` | Just configure .env |
| **Storage** | Your computer | Remote server |
| **Access** | Only you, only local | Team access, anywhere |
| **Authentication** | None needed | Username/password required |
| **Persistence** | Lost if you delete files | Always available |
| **Collaboration** | None | Full team collaboration |
| **Recommended For** | Quick testing | Production, team work |

---

## ✅ Verification Steps

### Check Connection

```python
import mlflow
from config import Config

# Setup
Config.setup_mlflow()

# Verify connection
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Expected: https://mlflow.shorthills.ai")

# Test by creating an experiment
try:
    exp = mlflow.get_experiment_by_name(Config.MLFLOW_EXPERIMENT_NAME)
    if exp:
        print(f"✅ Connected! Experiment ID: {exp.experiment_id}")
    else:
        exp_id = mlflow.create_experiment(Config.MLFLOW_EXPERIMENT_NAME)
        print(f"✅ Connected! Created new experiment: {exp_id}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

### Test with a Simple Run

```python
import mlflow
from config import Config

Config.setup_mlflow()

with mlflow.start_run(run_name="connection_test"):
    mlflow.log_param("test", "remote_connection")
    mlflow.log_metric("success", 1.0)
    print("✅ Test run logged to remote server!")
    print(f"View at: {Config.MLFLOW_TRACKING_URI}")
```

---

## 🎯 Best Practices for Remote Server

### 1. Use Unique Names

```python
# Always prefix with your identifier
YOUR_PREFIX = "john_doe"  # Change this!

# Experiments
mlflow.set_experiment(f"{YOUR_PREFIX}_genai_experiments")

# Prompts
mlflow.genai.register_prompt(
    name=f"{YOUR_PREFIX}_qa_prompt",
    template="..."
)

# Models
mlflow.pyfunc.log_model(
    registered_model_name=f"{YOUR_PREFIX}_genai_model"
)
```

### 2. Organize with Tags

```python
with mlflow.start_run():
    mlflow.set_tag("owner", "john_doe")
    mlflow.set_tag("team", "data_science")
    mlflow.set_tag("project", "customer_qa")
    # Your code...
```

### 3. Clean Up Artifacts Directory

Since artifacts go to remote server, you can periodically clean local cache:

```bash
# Local artifact cache (safe to delete)
rm -rf ~/.mlflow/artifacts/
```

### 4. Monitor Your Quota

Remote servers may have:
- Storage limits
- Rate limits
- Cost per artifact size

Check with your MLflow admin for limits.

---

## 🐛 Troubleshooting Remote Server

### Error: "Authentication failed"

**Solution:**
```bash
# Check credentials in .env
cat .env | grep MLFLOW_TRACKING

# Ensure no extra spaces
MLFLOW_TRACKING_USERNAME=myusername
MLFLOW_TRACKING_PASSWORD=mypassword
```

### Error: "Connection timeout"

**Solution:**
```bash
# Check if server is accessible
curl https://mlflow.shorthills.ai

# Check your internet connection
ping mlflow.shorthills.ai
```

### Error: "Experiment already exists"

**Solution:**
```bash
# Use a unique experiment name
MLFLOW_EXPERIMENT_NAME=yourname_unique_demo_v2
```

### Error: "Permission denied"

**Solution:**
- Verify your account has access
- Check with MLflow server admin
- Ensure you're not trying to modify others' experiments

### Error: "Artifact upload failed"

**Solution:**
```python
# Large artifacts may timeout
# Split into smaller chunks or compress

import gzip
# Compress before logging
mlflow.log_artifact("large_file.txt.gz")
```

---

## 📈 Monitoring Your Usage

### Check Your Experiments

```python
from mlflow import MlflowClient
from config import Config

Config.setup_mlflow()
client = MlflowClient()

# List your experiments
experiments = client.search_experiments()
for exp in experiments:
    if "yourname" in exp.name.lower():
        print(f"Experiment: {exp.name}")
        print(f"  ID: {exp.experiment_id}")
        print(f"  Location: {exp.artifact_location}")
```

### Count Your Runs

```python
runs = mlflow.search_runs(
    experiment_names=[Config.MLFLOW_EXPERIMENT_NAME]
)
print(f"Total runs: {len(runs)}")
print(f"Total metrics logged: {runs.shape[1]}")
```

---

## 🔒 Security Best Practices

### 1. Never Commit Credentials

```bash
# .env is already in .gitignore
# But verify:
cat .gitignore | grep .env

# Should see: .env
```

### 2. Use Environment Variables

```python
# Good - from environment
import os
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Bad - hardcoded
password = "my_password_123"  # Never do this!
```

### 3. Rotate Credentials Regularly

Ask your MLflow admin to rotate passwords periodically.

---

## 📚 Additional Resources

- **MLflow Tracking Docs**: https://mlflow.org/docs/latest/tracking.html
- **Remote Tracking**: https://mlflow.org/docs/latest/tracking.html#mlflow-tracking-servers
- **Authentication**: https://mlflow.org/docs/latest/auth/index.html

---

## ✨ Summary

When using `MLFLOW_TRACKING_URI=https://mlflow.shorthills.ai`:

✅ **All results stored remotely** (not on your computer)  
✅ **Team can view your experiments**  
✅ **No need to run local MLflow server**  
✅ **Results persist across sessions**  
✅ **Access from anywhere with internet**  
✅ **Professional, production-ready setup**  

⚠️ **Remember:**
- Use unique names (include your identifier)
- Keep credentials secure
- Check server quotas/limits
- Tag your experiments for organization

---

**Ready to use the remote server!** 🚀

Your results will appear at: https://mlflow.shorthills.ai

