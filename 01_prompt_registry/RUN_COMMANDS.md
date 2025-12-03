# Sequential Commands for Prompt Registry

## 📋 Quick Reference - Run Files in Order

### Prerequisites
```bash
# Start MLflow server (in separate terminal)
cd /home/shtlp_0170/Videos/ml_flow_features
mlflow ui --port 5000

# Or use the start script
python3 start_mlflow.py
```

---

## 🚀 Sequential Execution Order

### Step 1: Register Prompts (Including Long 1000+ Word Prompt) ⭐
```bash
cd /home/shtlp_0170/Videos/ml_flow_features/01_prompt_registry
python3 register_prompts.py
```

**What it does:**
- Registers 6 different prompts
- Includes 1 comprehensive prompt with 1000+ words
- Creates 2 versions of `qa_prompt_chat`
- Sets aliases (champion, baseline, challenger)

**Long Prompt Registered:**
- **Name**: `comprehensive_technical_writer`
- **Word Count**: 1000+ words
- **Purpose**: Detailed technical documentation generation
- **Use Case**: Complex system instructions for consistent output

---

### Step 2: Simple Fetch by Alias
```bash
python3 simple_fetch_by_alias.py
```

**What it does:**
- Loads prompts using aliases (champion, baseline)
- Demonstrates production pattern for prompt loading
- Shows how to use loaded prompts

---

### Step 3: Load and Use Prompts
```bash
python3 load_prompts.py
```

**What it does:**
- Loads prompts by version and alias
- Formats prompts with variables
- Uses prompts with LLMs

---

### Step 4: Use Long Comprehensive Prompt ⭐ NEW!
```bash
python3 use_long_prompt.py
```

**What it does:**
- Loads the 1000+ word comprehensive prompt
- Formats with specific input variables
- Generates documentation using Gemini
- Demonstrates multiple use cases

**Variables Used:**
- `topic`: The subject to document
- `audience_level`: Beginner/Intermediate/Advanced
- `doc_type`: Tutorial/API Reference/Guide
- `context`: Additional background information
- `desired_length`: Target length specification

**Output:**
- High-quality technical documentation
- Consistent formatting and structure
- Professional tone and style

---

### Step 5: Advanced Prompt Usage
```bash
python3 advanced_prompt_usage.py
```

**What it does:**
- Production workflow example
- Error handling and fallbacks
- Performance monitoring

---

### Step 6: Fetch from UI
```bash
python3 fetch_prompt_from_ui.py
```

**What it does:**
- Advanced fetching with debugging
- Inspect prompt metadata
- View template structure

---

### Step 7: Prompt Lifecycle Management
```bash
python3 prompt_lifecycle.py
```

**What it does:**
- Complete lifecycle: register → test → promote → deprecate
- Alias management
- Version control workflows

---

## 🎯 Quick Start (Just the Essentials)

If you want to see the long prompt feature quickly:

```bash
cd /home/shtlp_0170/Videos/ml_flow_features/01_prompt_registry

# Register all prompts (including the long one)
python3 register_prompts.py

# Use the long comprehensive prompt
python3 use_long_prompt.py
```

---

## 📊 View Results in MLflow UI

After running any script:

1. Open browser: **http://localhost:5000**
2. Click **"Prompts"** in the left sidebar
3. Find your prompts:
   - `qa_prompt_simple`
   - `qa_prompt_chat` (has 2 versions)
   - `summarization_prompt`
   - `sentiment_classifier`
   - `cot_reasoning_prompt`
   - `comprehensive_technical_writer` ⭐ (1000+ words)

4. Click on a prompt to see:
   - All versions
   - Aliases
   - Full template content
   - Tags and metadata
   - Commit messages

---

## 🔍 Verify Long Prompt

To verify the long prompt was registered correctly:

```bash
python3 -c "
import sys
sys.path.append('..')
from config import Config
import mlflow

Config.setup_mlflow()

# Load the long prompt
prompt = mlflow.genai.load_prompt('prompts:/comprehensive_technical_writer/latest')

# Check size
word_count = len(prompt.template.split())
char_count = len(prompt.template)

print(f'✅ Prompt: {prompt.name}')
print(f'✅ Version: {prompt.version}')
print(f'✅ Word Count: {word_count} words')
print(f'✅ Character Count: {char_count} characters')
print(f'✅ First 200 chars: {prompt.template[:200]}...')
"
```

---

## 💡 Use Cases for Long Prompts

### 1. Technical Documentation Generation
```bash
# The long prompt generates comprehensive documentation
python3 use_long_prompt.py
```

### 2. Consistent Content Creation
Long prompts ensure:
- Consistent style and tone
- Structured output format
- Quality standards enforcement
- Brand voice alignment

### 3. Complex Instruction Sets
Ideal for:
- Multi-step processes
- Detailed guidelines
- Few-shot learning examples
- Domain expertise embedding

### 4. Enterprise Applications
- API documentation generation
- Code review comments
- Technical blog posts
- Training materials
- Troubleshooting guides

---

## 📈 Benefits of Using Long Prompts

| Benefit | Impact |
|---------|--------|
| **Consistency** | Same output quality across all requests |
| **Context** | Comprehensive instructions reduce errors |
| **Reusability** | One prompt, multiple use cases |
| **Versioning** | Track complex instruction improvements |
| **Team Alignment** | Everyone uses same standards |
| **Quality** | Detailed specs = better outputs |
| **Scalability** | Generate at scale consistently |

---

## ⚡ Performance Considerations

**Token Usage:**
- Long prompts: Higher input tokens per request
- Example: 1000-word prompt ≈ 1,333 tokens

**Cost Example (Gemini 2.0 Flash):**
- Input: $0.075 per 1M tokens
- 1000-word prompt: ~$0.0001 per request
- Still cost-effective for quality improvement

**Context Window:**
- Gemini 2.0 Flash: 1,000,000 tokens
- Gemini 2.5 Pro: 2,000,000 tokens
- Plenty of room for long prompts + input data

---

## 🎓 Learning Path

### Beginner (15 minutes)
```bash
python3 register_prompts.py
python3 simple_fetch_by_alias.py
```

### Intermediate (30 minutes)
```bash
python3 register_prompts.py
python3 use_long_prompt.py
python3 advanced_prompt_usage.py
```

### Advanced (60 minutes)
```bash
# Run all files in sequence
python3 register_prompts.py
python3 simple_fetch_by_alias.py
python3 load_prompts.py
python3 use_long_prompt.py
python3 advanced_prompt_usage.py
python3 fetch_prompt_from_ui.py
python3 prompt_lifecycle.py
```

---

## 🐛 Troubleshooting

### Error: "Prompt not found"
```bash
# Solution: Register prompts first
python3 register_prompts.py
```

### Error: "GEMINI_API_KEY not found"
```bash
# Solution: Set up your .env file
cd /home/shtlp_0170/Videos/ml_flow_features
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Error: "Cannot connect to MLflow"
```bash
# Solution: Start MLflow server
mlflow ui --port 5000
# Or
python3 ../start_mlflow.py
```

---

## 📚 Additional Resources

- **README.md**: Comprehensive guide with code walkthroughs
- **../DOCUMENTATION.md**: Full project documentation
- **MLflow Prompt Docs**: https://mlflow.org/docs/latest/llms/prompt-engineering/

---

**Ready to start? Run the first command! 🚀**

```bash
python3 register_prompts.py
```


