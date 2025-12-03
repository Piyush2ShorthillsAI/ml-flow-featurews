# Prompt Evaluation with Gemini

## 📋 Overview

This folder contains scripts for evaluating prompts using MLflow's evaluation framework with **Gemini 2.0 Flash** model. All scripts have been updated to use Gemini instead of OpenAI.

---

## 🎯 What's Included

### 1. **evaluate_single_prompt.py**
- Evaluates a single prompt with custom and built-in scorers
- Uses Gemini 2.0 Flash for generation
- Includes custom scorers: conciseness, key concepts, response length
- Built-in scorers: Correctness, Guidelines

### 2. **compare_prompts.py**
- Compares multiple prompt versions side-by-side
- Evaluates different prompts with same scorers
- Identifies best performing prompt
- Provides recommendations

### 3. **evaluate_with_judges.py**
- Uses LLM-as-a-Judge evaluation
- Gemini judges evaluate quality, accuracy, helpfulness
- Provides detailed rationales for scores
- More nuanced than simple metrics

---

## 🔧 What Was Changed from OpenAI to Gemini

### API Changes
```python
# Before (OpenAI)
from openai import OpenAI
client = OpenAI(api_key=Config.OPENAI_API_KEY)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=formatted_messages
)

# After (Gemini)
import google.generativeai as genai
genai.configure(api_key=Config.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-001')
response = model.generate_content(full_prompt)
```

### Judge Models
```python
# Before
model="openai:/gpt-4o-mini"

# After  
model="gemini:/gemini-2.0-flash-001"
```

### Message Format Conversion
Gemini doesn't use the same chat format as OpenAI, so messages are converted:
```python
# Convert OpenAI chat format to Gemini format
if isinstance(formatted, list):
    system_msg = ""
    user_msg = ""
    for msg in formatted:
        if msg.get("role") == "system":
            system_msg = msg.get("content", "")
        elif msg.get("role") == "user":
            user_msg = msg.get("content", "")
    
    full_prompt = f"{system_msg}\n\n{user_msg}" if system_msg else user_msg
```

---

## 🚀 How to Run

### Prerequisites
```bash
# 1. Make sure MLflow server is running
mlflow ui --port 5000

# 2. Register prompts first (if not already done)
cd ../01_prompt_registry
python3 register_prompts.py
cd ../02_prompt_evaluation
```

### Run Scripts

#### Evaluate Single Prompt
```bash
python3 evaluate_single_prompt.py
```

**What it does:**
- Loads `qa_prompt_chat@latest` from registry
- Evaluates on sample Q&A dataset
- Uses 5 different scorers
- Logs results to MLflow

**Expected Output:**
```
1️⃣  Setting up prediction function...
✅ Prediction function ready

2️⃣  Preparing evaluation dataset...
✅ Created dataset with 4 samples

3️⃣  Defining evaluation scorers...
✅ Defined 5 scorers

4️⃣  Running evaluation...
✅ Evaluation complete!

📊 Metrics:
   correctness/score: 0.8750
   is_concise/score: 0.7500
   ...
```

#### Compare Prompt Versions
```bash
python3 compare_prompts.py
```

**What it does:**
- Compares 3 prompt versions:
  - `qa_prompt_simple@latest`
  - `qa_prompt_chat/1` (version 1)
  - `qa_prompt_chat@latest` (version 2)
- Evaluates each with same dataset
- Shows best performer

**Expected Output:**
```
1️⃣  Setting up prediction functions...
✅ Will compare 3 prompt versions

3️⃣  Evaluating each prompt version...
   ✅ Simple Prompt complete
   ✅ Chat V1 complete
   ✅ Chat V2 (Latest) complete

🏆 Best Performing Prompts:
   correctness: Chat V2 (Latest)
   ...

🥇 Overall Best Prompt: Chat V2 (Latest)
```

#### Evaluate with LLM Judges
```bash
python3 evaluate_with_judges.py
```

**What it does:**
- Creates 4 Gemini-based judges
- Each judge evaluates different aspects
- Provides detailed rationales
- More comprehensive evaluation

**Expected Output:**
```
2️⃣  Creating LLM judge scorers...
✅ Created quality judge
✅ Created technical accuracy judge
✅ Created helpfulness judge
✅ Created content match judge

4️⃣  Running evaluation with LLM judges...
   ⏳ This will take longer as judges are LLM-based...

🤖 LLM Judge Insights:
   Quality Distribution:
      excellent: 2 samples
      good: 2 samples
```

---

## 📊 View Results in MLflow UI

After running any script:

1. **Open MLflow UI**: http://localhost:5000
2. **Experiments Tab**:
   - Find your experiment (default: `gemini_qa_demo`)
   - Click on the experiment
   - See runs: `prompt_evaluation_single`, `eval_Simple_Prompt`, etc.
3. **Run Details**:
   - Parameters: prompt_name, model, temperature
   - Metrics: Various scorer results
   - Artifacts: Evaluation tables, results

---

## 🎯 Evaluation Scorers Explained

### Custom Scorers

#### 1. **is_concise**
```python
@scorer
def is_concise(outputs: str) -> bool:
    """Check if response is under 100 words"""
    return len(outputs.split()) <= 100
```

#### 2. **contains_key_concepts**
```python
@scorer
def contains_key_concepts(outputs: str, expectations: dict) -> Feedback:
    """Check if response includes expected concepts"""
    # Returns score 0.0-1.0 based on concept coverage
```

#### 3. **response_length_score**
```python
@scorer  
def response_length_score(outputs: str) -> Feedback:
    """Score based on optimal length (20-80 words)"""
    # Returns 1.0 for optimal, 0.5 for too short, 0.7 for long
```

#### 4. **clarity_score**
```python
@scorer
def clarity_score(outputs: str) -> Feedback:
    """Score based on clarity (examples, structure)"""
    # Checks for examples and clear structure
```

### Built-in Scorers

#### 1. **Correctness**
```python
Correctness(name="factual_correctness")
# Uses LLM to check factual accuracy
```

#### 2. **Guidelines**
```python
Guidelines(
    name="professional_tone",
    guidelines="Response should be professional and clear"
)
# Evaluates against custom guidelines
```

### LLM Judge Scorers

#### 1. **Quality Judge**
- Evaluates overall quality
- Considers accuracy, completeness, clarity
- Rates: excellent, good, fair, poor

#### 2. **Technical Accuracy Judge**
- Checks technical correctness
- Verifies terminology usage
- Rates: accurate, mostly_accurate, partially_accurate, inaccurate

#### 3. **Helpfulness Judge**
- Assesses practical value
- Checks if question is addressed
- Rates: very_helpful, helpful, somewhat_helpful, not_helpful

#### 4. **Content Match Judge**
- Compares with expected output
- Checks key concept coverage
- Rates: matches, mostly_matches, partially_matches, no_match

---

## 💡 Use Cases

### 1. **Prompt Optimization**
```bash
# Compare different prompt versions
python3 compare_prompts.py

# Identify best performer
# Promote to production using alias
```

### 2. **Quality Assurance**
```bash
# Evaluate before deployment
python3 evaluate_single_prompt.py

# Check if metrics meet thresholds
# Ensure quality standards
```

### 3. **A/B Testing**
```bash
# Evaluate multiple candidates
python3 compare_prompts.py

# Make data-driven decisions
# Track improvements over time
```

### 4. **Regression Testing**
```bash
# Re-evaluate after changes
python3 evaluate_single_prompt.py

# Compare with baseline metrics
# Ensure no degradation
```

---

## 🔍 Understanding Evaluation Results

### Metrics Interpretation

**Scores (0.0 - 1.0):**
- `1.0` = Perfect/Excellent
- `0.8-0.9` = Very Good
- `0.6-0.7` = Good
- `0.4-0.5` = Fair
- `< 0.4` = Needs Improvement

**Boolean Scores:**
- `1.0` = True/Pass
- `0.0` = False/Fail

### Sample Results Table

| Sample | is_concise | correctness | clarity | overall |
|--------|------------|-------------|---------|---------|
| 1      | 1.0        | 0.85        | 0.80    | 0.88    |
| 2      | 0.0        | 0.90        | 0.75    | 0.55    |
| 3      | 1.0        | 0.95        | 0.85    | 0.93    |
| 4      | 1.0        | 0.80        | 0.70    | 0.83    |

**Average:** 0.80

---

## ⚙️ Configuration

### Model Settings

Edit in script or config.py:
```python
# Model
model = 'gemini-2.0-flash-001'

# Temperature (0.0-1.0)
temperature = 0.7  # Default

# Max tokens
max_output_tokens = 150  # For concise responses
```

### Scorer Thresholds

Customize thresholds in scorers:
```python
# Example: Adjust conciseness threshold
@scorer
def is_concise(outputs: str) -> bool:
    return len(outputs.split()) <= 80  # Changed from 100
```

---

## 🐛 Troubleshooting

### Issue: "GEMINI_API_KEY not set"

**Solution:**
```bash
# Check .env file
cat ../.env | grep GEMINI

# Should see:
# GEMINI_API_KEY=your_key_here

# If missing, add it:
echo "GEMINI_API_KEY=your_actual_key" >> ../.env
```

### Issue: "Prompt not found"

**Solution:**
```bash
# Register prompts first
cd ../01_prompt_registry
python3 register_prompts.py
cd ../02_prompt_evaluation
```

### Issue: "Judge model not found"

**Cause:** MLflow might not recognize `gemini:/` prefix

**Solution:**
The scripts are configured correctly. If you see errors, ensure MLflow version supports Gemini judges (MLflow >= 3.4.0):
```bash
pip install --upgrade mlflow
```

### Issue: Evaluation takes too long

**Cause:** LLM judges make multiple API calls

**Workaround:**
```bash
# Use smaller dataset for testing
# Edit the script and reduce samples:
eval_data = create_sample_qa_data()[:2]  # Only 2 samples
```

---

## 📈 Performance

### Response Times

| Script | Samples | Scorers | Time |
|--------|---------|---------|------|
| evaluate_single_prompt | 4 | 5 | ~15-20s |
| compare_prompts | 4 × 3 | 3 | ~40-60s |
| evaluate_with_judges | 4 | 6 (4 LLM) | ~60-90s |

**Note:** LLM judges are slower but provide richer insights

### Cost Estimation (Gemini 2.0 Flash)

- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens

**Example:**
- 4 samples × 5 scorers ≈ 20 API calls
- Average 200 tokens per call
- Total: ~4,000 tokens
- Cost: < $0.01 per evaluation

---

## 🎓 Best Practices

### 1. **Start Simple**
```bash
# First, evaluate single prompt
python3 evaluate_single_prompt.py

# Understand metrics
# Then compare versions
python3 compare_prompts.py
```

### 2. **Use Appropriate Scorers**
- **Custom scorers**: Fast, deterministic, specific rules
- **Built-in scorers**: Balance of speed and quality
- **LLM judges**: Slow but nuanced, for final evaluation

### 3. **Version Control Prompts**
```bash
# Always use aliases
prompt = mlflow.genai.load_prompt("prompts:/name@champion")

# Track versions in evaluation
mlflow.log_param("prompt_version", prompt.version)
```

### 4. **Iterate Based on Results**
```
1. Evaluate → 2. Identify weaknesses → 3. Improve prompt → 4. Re-evaluate
```

### 5. **Set Quality Gates**
```python
# Example threshold
if results.metrics['correctness/score'] < 0.8:
    print("❌ Quality gate failed - improve prompt")
else:
    print("✅ Ready for production")
```

---

## 📚 Additional Resources

- **MLflow Evaluation**: https://mlflow.org/docs/latest/llms/llm-evaluate/
- **Custom Scorers**: https://mlflow.org/docs/latest/llms/custom-metrics/
- **LLM Judges**: https://mlflow.org/docs/latest/llms/llm-judge/
- **Gemini API**: https://ai.google.dev/docs

---

## ✅ Summary

| Feature | Status |
|---------|--------|
| **Gemini Integration** | ✅ Complete |
| **Custom Scorers** | ✅ 4 scorers |
| **Built-in Scorers** | ✅ 2 scorers |
| **LLM Judges** | ✅ 4 judges (Gemini-based) |
| **Prompt Comparison** | ✅ Side-by-side |
| **MLflow Logging** | ✅ Full tracking |
| **Tracing** | ✅ Enabled |

---

**🚀 Ready to start? Run:**
```bash
python3 evaluate_single_prompt.py
```

**💡 View results:** http://localhost:5000

