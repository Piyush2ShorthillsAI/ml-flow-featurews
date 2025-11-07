# Getting Started with MLflow Features Demo

This guide will help you quickly get started with the MLflow Features demonstration project.

## Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
cd /home/shtlp_0170/Videos/ml_flow_features
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

**Required:** Set at least `OPENAI_API_KEY` in your `.env` file:
```bash
OPENAI_API_KEY=sk-your-key-here
```

### 3. Start MLflow Server

Open a **new terminal** and run:

```bash
cd /home/shtlp_0170/Videos/ml_flow_features
mlflow ui --port 5000
```

Keep this terminal running. Access the UI at: http://localhost:5000

### 4. Run Your First Example

In your original terminal:

```bash
# Register prompts
python 01_prompt_registry/register_prompts.py

# Evaluate a prompt
python 02_prompt_evaluation/evaluate_single_prompt.py
```

View results in the MLflow UI at http://localhost:5000

## Step-by-Step Learning Path

### Beginner Track (30 minutes)

1. **Prompt Management** (10 min)
   ```bash
   python 01_prompt_registry/register_prompts.py
   python 01_prompt_registry/load_prompts.py
   ```

2. **Prompt Evaluation** (10 min)
   ```bash
   python 02_prompt_evaluation/evaluate_single_prompt.py
   ```

3. **Tracing** (10 min)
   ```bash
   python 05_tracing/openai_tracing.py
   ```

### Intermediate Track (1 hour)

4. **Model Registry** (20 min)
   ```bash
   python 03_model_registry/register_genai_model.py
   python 03_model_registry/load_and_predict.py
   ```

5. **Custom Scorers** (20 min)
   ```bash
   python 04_evaluation_framework/custom_scorers.py
   python 04_evaluation_framework/llm_judges.py
   ```

6. **Evaluation Datasets** (20 min)
   ```bash
   python 06_evaluation_datasets/create_from_data.py
   ```

### Advanced Track (1 hour)

7. **Custom Tracing** (20 min)
   ```bash
   python 05_tracing/custom_tracing.py
   ```

8. **ResponsesAgent** (20 min)
   ```bash
   python 07_responses_agent/simple_agent.py
   ```

9. **End-to-End Pipeline** (20 min)
   ```bash
   python 08_end_to_end/full_pipeline.py
   ```

## Quick Feature Reference

### Prompt Registry
```python
import mlflow

# Register
prompt = mlflow.genai.register_prompt(
    name="my_prompt",
    template="Answer: {{question}}"
)

# Load
prompt = mlflow.genai.load_prompt("prompts:/my_prompt@latest")
```

### Model Registration
```python
# Register GenAI model
mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=my_model,
    registered_model_name="my_genai_model"
)

# Load
model = mlflow.pyfunc.load_model("models:/my_genai_model/1")
```

### Evaluation
```python
from mlflow.genai import scorer

@scorer
def my_scorer(outputs: str) -> bool:
    return len(outputs.split()) <= 100

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=my_predict_fn,
    scorers=[my_scorer]
)
```

### Tracing
```python
import mlflow

# Auto-trace OpenAI
mlflow.openai.autolog()

# Custom trace
@mlflow.trace
def my_function(x):
    return x * 2
```

## Troubleshooting

### Issue: "OPENAI_API_KEY not set"
**Solution:** Make sure you've created `.env` file and set the API key.

### Issue: "Could not connect to MLflow server"
**Solution:** Start the MLflow server: `mlflow ui --port 5000`

### Issue: "Prompt not found"
**Solution:** Run the prompt registration script first:
```bash
python 01_prompt_registry/register_prompts.py
```

### Issue: Import errors
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

## Common Workflows

### Workflow 1: Evaluate Multiple Prompts
```bash
# 1. Register different prompt versions
python 01_prompt_registry/register_prompts.py

# 2. Compare them
python 02_prompt_evaluation/compare_prompts.py
```

### Workflow 2: Model Development Pipeline
```bash
# 1. Register model
python 03_model_registry/register_genai_model.py

# 2. Evaluate model
python 02_prompt_evaluation/evaluate_single_prompt.py

# 3. View traces
# Check MLflow UI
```

### Workflow 3: Production Deployment
```bash
# 1. Register and version model
python 03_model_registry/register_genai_model.py

# 2. Promote to production
python 03_model_registry/model_versioning.py

# 3. Load production model
model = mlflow.pyfunc.load_model("models:/genai_qa_model/Production")
```

## Next Steps

1. **Explore MLflow UI**: Navigate to http://localhost:5000 to see:
   - Experiments and runs
   - Registered prompts
   - Registered models
   - Traces
   - Evaluation results

2. **Customize Examples**: Modify the examples for your use case

3. **Read Documentation**: Check out the main [README.md](README.md)

4. **Review Context Files**: See `mlflowcontext/` for detailed MLflow documentation

## Resources

- **MLflow UI**: http://localhost:5000
- **MLflow Docs**: https://mlflow.org/docs/latest/
- **OpenAI Docs**: https://platform.openai.com/docs

## Support

For issues or questions:
1. Check the [README.md](README.md)
2. Review example code comments
3. Check MLflow documentation
4. Review context files in `mlflowcontext/`

---

**Happy Learning! 🚀**

