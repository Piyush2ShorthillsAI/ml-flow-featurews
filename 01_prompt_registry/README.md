# Prompt Registry - Version Control for LLM Prompts

## 📋 Overview

The Prompt Registry provides **Git-like version control for LLM prompts**, enabling teams to manage, version, and deploy prompts systematically.

---

## 🎯 Use Cases

### 1. **Prompt Versioning**
- Track changes to prompts over time
- Rollback to previous versions if needed
- Compare different prompt iterations

### 2. **A/B Testing**
- Test multiple prompt variants
- Use aliases to manage test groups
- Measure which prompts perform better

### 3. **Team Collaboration**
- Centralized prompt storage
- Share prompts across team
- Consistent prompts across environments

### 4. **Production Safety**
- Separate development from production prompts
- Gradual rollout of new prompts
- Quick rollback capability

---

## 📁 Files in This Directory

### 1. `register_prompts.py` ⭐

**Purpose**: Register and version prompts in MLflow (includes 1000+ word comprehensive prompt)

**What it does**:
1. Creates new prompt entries in MLflow
2. Assigns version numbers automatically
3. Sets aliases for easy access
4. Stores metadata (tags, commit messages)

**Code Walkthrough**:

```python
# Import MLflow GenAI module
import mlflow

# Step 1: Register a simple text prompt
simple_prompt = mlflow.genai.register_prompt(
    name="qa_prompt_simple",           # Unique identifier
    template="Answer: {{question}}",   # Template with variables
    commit_message="Initial version",  # Description of change
    tags={"type": "qa"}                # Metadata for organization
)
# Returns: Prompt object with .name, .version, .template

# Step 2: Register a chat-style prompt (list format)
chat_template = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    },
    {
        "role": "user",
        "content": "Question: {{question}}"
    }
]

chat_prompt = mlflow.genai.register_prompt(
    name="qa_prompt_chat",
    template=chat_template,  # List of message dictionaries
    commit_message="Chat-style Q&A prompt",
    tags={"type": "chat", "version": "v1"}
)

# Step 3: Create a new version (same name = new version)
improved_template = [
    {
        "role": "system",
        "content": "You are a knowledgeable AI assistant. Provide detailed answers."
    },
    {
        "role": "user",
        "content": "Question: {{question}}\n\nPlease provide a comprehensive answer."
    }
]

chat_prompt_v2 = mlflow.genai.register_prompt(
    name="qa_prompt_chat",  # Same name!
    template=improved_template,
    commit_message="Improved with detailed instructions",
    tags={"type": "chat", "version": "v2"}
)
# Auto-increments version: v1 → v2

# Step 4: Set aliases (named pointers to versions)
mlflow.genai.set_prompt_alias(
    name="qa_prompt_chat",
    alias="champion",        # Best performing version
    version=chat_prompt_v2.version
)

mlflow.genai.set_prompt_alias(
    name="qa_prompt_chat",
    alias="baseline",        # Stable reference
    version=chat_prompt.version
)

mlflow.genai.set_prompt_alias(
    name="qa_prompt_chat",
    alias="challenger",      # Version being tested
    version=chat_prompt_v2.version
)
```

**What happens internally**:
1. MLflow checks if prompt name exists
2. If yes: increments version number
3. If no: creates new entry with version 1
4. Stores template as JSON in database
5. Creates alias → version mappings

**Run it**:
```bash
cd 01_prompt_registry
python3 register_prompts.py
```

**Output**:
```
1️⃣  Registering a simple text prompt...
✅ Registered prompt: qa_prompt_simple
   Version: 1
   Template: Answer the following question: {{question}}

2️⃣  Registering a chat-style prompt...
✅ Registered chat prompt: qa_prompt_chat
   Version: 1

3️⃣  Creating version 2 of the prompt...
✅ Created new version: 2
   Previous version: 1

5️⃣  Setting aliases for prompt versions...
✅ Set 'champion' alias to version 2
✅ Set 'baseline' alias to version 1
✅ Set 'challenger' alias to version 2
```

---

### 2. `simple_fetch_by_alias.py`

**Purpose**: Load prompts by alias or version (production pattern)

**What it does**:
1. Loads prompts from MLflow registry
2. Supports loading by alias (champion, baseline)
3. Supports loading by specific version
4. Formats templates with variables

**Code Walkthrough**:

```python
import mlflow
from config import Config

# Setup MLflow connection
Config.setup_mlflow()

# Method 1: Load by alias (recommended for production)
prompt = mlflow.genai.load_prompt(
    name="qa_prompt_chat",
    alias="champion"  # Always gets current champion version
)
# Returns: Prompt object

# Access template
print(f"Template: {prompt.template}")
# Output: List of message dictionaries

# Method 2: Load by specific version (for testing)
prompt_v1 = mlflow.genai.load_prompt(
    name="qa_prompt_chat",
    version=1
)

# Method 3: Format template with variables
formatted = prompt.template.copy()
formatted[1]["content"] = formatted[1]["content"].replace(
    "{{question}}", 
    "What is machine learning?"
)

# Now ready to send to LLM!
response = call_gemini(formatted)
```

**What happens internally**:
1. Queries MLflow server: `SELECT * FROM prompt_registry WHERE name=? AND alias=?`
2. Resolves alias to version number
3. Fetches template from database
4. Returns Prompt object with `.template`, `.version`, `.name`

**Run it**:
```bash
cd 01_prompt_registry

# Set environment variables (optional)
export MLFLOW_PROMPT_NAME="qa_prompt_chat"
export MLFLOW_PROMPT_ALIAS="champion"

python3 simple_fetch_by_alias.py
```

**Output**:
```
Loading prompt: qa_prompt_chat (alias: champion)
✅ Loaded prompt version: 2

Template structure:
- Role: system
  Content: You are a knowledgeable AI assistant...
- Role: user
  Content: Question: {{question}}...

Formatting with question: "What is artificial intelligence?"
✅ Formatted prompt ready for LLM
```

---

### 3. `fetch_prompt_from_ui.py`

**Purpose**: Advanced prompt fetching with debugging and inspection

**What it does**:
1. Implements 6 different methods to extract templates
2. Full object inspection (attributes, methods)
3. Detailed debugging output
4. Handles edge cases

**Code Walkthrough**:

```python
class PromptFetcher:
    """Advanced prompt fetching with multiple extraction methods"""
    
    def __init__(self, prompt_name, alias=None, version=None):
        self.prompt_name = prompt_name
        self.alias = alias
        self.version = version
        self.prompt_obj = None
    
    def load_prompt(self):
        """Load prompt from MLflow"""
        if self.alias:
            # Method 1: By alias
            self.prompt_obj = mlflow.genai.load_prompt(
                name=self.prompt_name,
                alias=self.alias
            )
        elif self.version:
            # Method 2: By version
            self.prompt_obj = mlflow.genai.load_prompt(
                name=self.prompt_name,
                version=self.version
            )
        else:
            raise ValueError("Provide either alias or version")
    
    def extract_template_method_1(self):
        """Direct .template attribute access"""
        return self.prompt_obj.template
    
    def extract_template_method_2(self):
        """Using vars() to get all attributes"""
        return vars(self.prompt_obj).get('template')
    
    def extract_template_method_3(self):
        """Using __dict__ directly"""
        return self.prompt_obj.__dict__.get('template')
    
    def extract_template_method_4(self):
        """Using getattr with fallback"""
        return getattr(self.prompt_obj, 'template', None)
    
    def extract_template_method_5(self):
        """Check for get_template() method"""
        if hasattr(self.prompt_obj, 'get_template'):
            return self.prompt_obj.get_template()
        return None
    
    def extract_template_method_6(self):
        """Inspect private attributes"""
        for attr in dir(self.prompt_obj):
            if 'template' in attr.lower():
                return getattr(self.prompt_obj, attr, None)
        return None
    
    def inspect_object(self):
        """Full object inspection"""
        print("\n" + "="*60)
        print("FULL OBJECT INSPECTION")
        print("="*60)
        
        # Type
        print(f"\n1️⃣  Object Type: {type(self.prompt_obj)}")
        
        # All attributes
        print(f"\n2️⃣  All Attributes:")
        for attr in dir(self.prompt_obj):
            if not attr.startswith('_'):
                print(f"   - {attr}")
        
        # __dict__ contents
        print(f"\n3️⃣  Object Dictionary:")
        for key, value in self.prompt_obj.__dict__.items():
            print(f"   - {key}: {type(value).__name__}")
        
        # Try all extraction methods
        print(f"\n4️⃣  Extraction Method Results:")
        methods = [
            self.extract_template_method_1,
            self.extract_template_method_2,
            self.extract_template_method_3,
            self.extract_template_method_4,
            self.extract_template_method_5,
            self.extract_template_method_6
        ]
        
        for i, method in enumerate(methods, 1):
            try:
                result = method()
                status = "✅ SUCCESS" if result else "❌ NULL"
                print(f"   Method {i}: {status}")
            except Exception as e:
                print(f"   Method {i}: ❌ ERROR - {e}")

# Usage
fetcher = PromptFetcher(
    prompt_name="qa_prompt_chat",
    alias="champion"
)

fetcher.load_prompt()
fetcher.inspect_object()
template = fetcher.extract_template_method_1()
```

**When to use**:
- Debugging prompt loading issues
- Understanding MLflow Prompt object structure
- Testing new MLflow versions
- Educational purposes

**Run it**:
```bash
cd 01_prompt_registry
python3 fetch_prompt_from_ui.py
```

---

### 4. `advanced_prompt_usage.py`

**Purpose**: Production workflow example with Gemini integration

**What it does**:
1. Fetches prompts from registry
2. Formats with input variables
3. Calls Gemini API
4. Logs results to MLflow
5. Compares multiple prompt versions

**Code Walkthrough**:

```python
def generate_with_prompt(prompt_alias, question):
    """
    Complete workflow: Load prompt → Format → Generate → Log
    """
    
    # Start MLflow run for tracking
    with mlflow.start_run(run_name=f"generate_{prompt_alias}"):
        
        # 1. Load prompt from registry
        prompt = mlflow.genai.load_prompt(
            name="qa_prompt_chat",
            alias=prompt_alias
        )
        mlflow.log_param("prompt_alias", prompt_alias)
        mlflow.log_param("prompt_version", prompt.version)
        
        # 2. Format prompt with question
        formatted = format_prompt(prompt.template, question)
        
        # 3. Call Gemini API
        start_time = time.time()
        response = call_gemini_api(formatted)
        latency = time.time() - start_time
        
        # 4. Log results
        mlflow.log_metric("latency_ms", latency * 1000)
        mlflow.log_metric("response_length", len(response))
        mlflow.log_param("question", question)
        mlflow.log_text(response, "response.txt")
        
        return response

# Compare multiple prompt versions
question = "What is machine learning?"

# Test champion
response_champion = generate_with_prompt("champion", question)

# Test baseline
response_baseline = generate_with_prompt("baseline", question)

# Test challenger
response_challenger = generate_with_prompt("challenger", question)

# Results logged to MLflow for comparison
```

**What happens internally**:
1. Creates separate MLflow run for each prompt version
2. Tracks which prompt version was used
3. Measures latency and other metrics
4. Stores responses as artifacts
5. Enables side-by-side comparison in UI

**Run it**:
```bash
cd 01_prompt_registry
python3 advanced_prompt_usage.py
```

**Output**:
```
Testing prompt: champion
✅ Generated response (latency: 450ms)

Testing prompt: baseline
✅ Generated response (latency: 380ms)

Testing prompt: challenger
✅ Generated response (latency: 520ms)

📊 Compare results in MLflow UI
```

---

### 5. `load_prompts.py`

**Purpose**: Basic prompt loading examples

**What it does**:
- Load prompts by name
- Load prompts by version
- List all available prompts

---

### 6. `prompt_lifecycle.py`

**Purpose**: Manage prompt lifecycle and transitions

**What it does**:
- Update prompt descriptions
- Move aliases between versions
- Archive old prompts

---

## 🔄 Typical Workflow

### 1. **Development Phase**

```bash
# 1. Create and register initial prompt
python3 register_prompts.py

# 2. Test the prompt
python3 simple_fetch_by_alias.py

# 3. Iterate and create v2, v3, etc.
# Edit register_prompts.py to add new versions
python3 register_prompts.py
```

### 2. **Testing Phase**

```bash
# Compare different versions
python3 advanced_prompt_usage.py

# Analyze results in MLflow UI
# http://localhost:5000 or https://mlflow.shorthills.ai
```

### 3. **Production Deployment**

```python
# In your application code:
from mlflow import genai

# Always use alias (not hardcoded version!)
prompt = genai.load_prompt("qa_prompt_chat", alias="champion")

# Format and use
formatted = format(prompt.template, question=user_question)
response = call_llm(formatted)
```

### 4. **Version Updates**

```bash
# Register new version
python3 register_prompts.py  # Creates v4

# Update alias to new version (no code changes needed!)
mlflow.genai.set_prompt_alias(
    name="qa_prompt_chat",
    alias="champion",
    version=4  # Now points to v4
)

# All production code automatically uses v4!
```

---

## 💡 Best Practices

### ✅ DO

1. **Use Descriptive Names**
   ```python
   # Good
   "customer_support_friendly_tone"
   
   # Bad
   "prompt_1"
   ```

2. **Write Clear Commit Messages**
   ```python
   # Good
   commit_message="Improved empathy, reduced technical jargon"
   
   # Bad
   commit_message="update"
   ```

3. **Use Aliases for Deployment**
   ```python
   # Good: Decouples code from versions
   load_prompt("qa_prompt", alias="champion")
   
   # Bad: Hardcoded version
   load_prompt("qa_prompt", version=3)
   ```

4. **Tag Prompts with Metadata**
   ```python
   tags={
       "use_case": "customer_support",
       "tone": "friendly",
       "language": "english",
       "tested": "true"
   }
   ```

5. **Version Incrementally**
   - Make small, testable changes
   - Test before updating aliases
   - Keep old versions for rollback

### ❌ DON'T

1. **Don't Use Reserved Alias Names**
   ```python
   # Reserved by MLflow (will error)
   alias="latest"  # ❌
   alias="production"  # ❌
   alias="staging"  # ❌
   
   # Use instead
   alias="champion"  # ✅
   alias="baseline"  # ✅
   alias="challenger"  # ✅
   ```

2. **Don't Hardcode Prompts in Code**
   ```python
   # Bad
   prompt = "Answer this question: {question}"
   
   # Good
   prompt = mlflow.genai.load_prompt("qa_prompt", alias="champion")
   ```

3. **Don't Skip Testing**
   - Always test new versions before aliasing
   - Use challenger alias for testing
   - Compare metrics before promoting

4. **Don't Delete Old Versions**
   - Keep for rollback capability
   - Historical reference
   - A/B test comparisons

---

## 🐛 Troubleshooting

### Issue 1: Prompt Not Found

**Error**: `MLflowException: Prompt 'qa_prompt_chat' not found`

**Solution**:
```bash
# 1. Register the prompt first
python3 register_prompts.py

# 2. Check prompt exists
python3 -c "import mlflow; print(mlflow.genai.list_prompts())"
```

---

### Issue 2: Alias Not Found

**Error**: `MLflowException: Alias 'champion' not found`

**Solution**:
```bash
# 1. Check aliases
python3 -c "import mlflow; print(mlflow.genai.list_prompt_aliases('qa_prompt_chat'))"

# 2. Set alias
python3 register_prompts.py  # Sets aliases automatically
```

---

### Issue 3: Reserved Alias Name

**Error**: `INVALID_PARAMETER_VALUE: 'latest' alias name is reserved`

**Solution**:
```python
# Don't use
alias="latest"  # Reserved

# Use instead
alias="champion"  # Custom alias name
```

---

### Issue 4: Template Format Issues

**Problem**: Variables not being replaced

**Solution**:
```python
# Ensure template uses {{variable}} syntax
template = "Question: {{question}}"  # ✅

# Not
template = "Question: {question}"  # ❌
```

---

## 📊 Viewing Results

### MLflow UI

1. Navigate to MLflow UI: `http://localhost:5000` or your tracking URI
2. Click "Prompts" in sidebar
3. Find your prompt name: `qa_prompt_chat`
4. View:
   - All versions
   - Aliases (champion, baseline, challenger)
   - Templates
   - Commit messages
   - Tags

### Programmatic Access

```python
import mlflow

# List all prompts
prompts = mlflow.genai.list_prompts()
print(prompts)

# List versions of a prompt
versions = mlflow.genai.list_prompt_versions("qa_prompt_chat")
print(versions)

# List aliases
aliases = mlflow.genai.list_prompt_aliases("qa_prompt_chat")
print(aliases)
```

---

## 📝 Working with Long Prompts (1000+ words)

### 7. `use_long_prompt.py`

**Purpose**: Demonstrate how to register and use comprehensive prompts longer than 1000 words

**What it does**:
1. Loads a comprehensive technical documentation writer prompt (1000+ words)
2. Formats the prompt with specific input variables
3. Uses the long prompt with Gemini to generate documentation
4. Demonstrates multiple use cases for the same comprehensive prompt

**When to Use Long Prompts**:

Long prompts (1000+ words) are ideal for:
- **Detailed System Instructions**: Comprehensive guidelines for LLM behavior
- **Few-Shot Learning**: Multiple examples showing desired output format
- **Complex Documentation**: Generating consistent, high-quality documentation
- **Style Guides**: Enforcing specific writing styles and standards
- **Multi-Step Processes**: Detailed workflows with decision trees
- **Domain Expertise**: Embedding extensive domain knowledge

**Example Use Case**:

The `comprehensive_technical_writer` prompt registered in `register_prompts.py` contains:
- 1000+ words of detailed instructions
- Clear documentation standards
- Structure guidelines
- Code quality requirements
- Tone and voice specifications
- Quality checklists

**Run the Example**:

```bash
# First, register the long prompt
python register_prompts.py

# Then use it to generate documentation
python use_long_prompt.py
```

**Code Example**:

```python
import mlflow

# Load the long comprehensive prompt
prompt = mlflow.genai.load_prompt("prompts:/comprehensive_technical_writer/latest")

# Check the size
word_count = len(prompt.template.split())
print(f"Word count: {word_count} words")  # 1000+ words

# Format with variables
formatted = prompt.format(
    topic="MLflow Prompt Registry",
    audience_level="Intermediate",
    doc_type="API Reference",
    context="Developers versioning LLM prompts",
    desired_length="Comprehensive"
)

# Use with your LLM
response = model.generate_content(formatted)
```

**Benefits of Long Comprehensive Prompts**:

| Benefit | Description |
|---------|-------------|
| **Consistency** | Same detailed instructions ensure uniform output quality |
| **Context** | Comprehensive context reduces ambiguity and errors |
| **Reusability** | One prompt serves multiple related tasks |
| **Versioning** | Track improvements to complex instructions over time |
| **Team Alignment** | Entire team uses identical standards and guidelines |
| **Quality** | Detailed specifications lead to higher quality outputs |
| **Scalability** | Generate content at scale with consistent quality |

**Performance Considerations**:

- **Token Usage**: Long prompts consume more input tokens
- **Latency**: Slightly longer processing time for initial input
- **Cost**: Higher per-request cost due to increased token count
- **Context Windows**: Ensure your LLM can handle the prompt size (Gemini 2.0 Flash: 1M tokens)

**Best Practices**:

1. **Structure Clearly**: Use sections and headers in long prompts
2. **Include Examples**: Show desired output format within the prompt
3. **Version Control**: Use MLflow to track changes to complex prompts
4. **Test Thoroughly**: Validate output quality across different inputs
5. **Optimize**: Remove unnecessary verbosity while maintaining clarity
6. **Document**: Add tags and commit messages explaining prompt purpose

**Real-World Applications**:

- **Technical Documentation**: Generate API docs, tutorials, and guides
- **Content Creation**: Blog posts, articles with specific style requirements
- **Code Generation**: Complex codebases with architectural patterns
- **Data Analysis**: Comprehensive analytical reports with specific structure
- **Customer Support**: Detailed troubleshooting responses with standard format

---

## 🚀 Next Steps

1. **Explore Prompt Evaluation**: Go to `../02_prompt_evaluation/`
2. **Integrate with Models**: Go to `../03_model_registry/`
3. **Read Main Docs**: `../DOCUMENTATION.md`

---

## 📚 Additional Resources

- [MLflow GenAI Docs](https://mlflow.org/docs/latest/llms/genai/index.html)
- [MLflow Prompt Engineering Guide](https://mlflow.org/docs/latest/llms/prompt-engineering/index.html)
- Main project documentation: `../DOCUMENTATION.md`

---

**Questions?** Check the main `DOCUMENTATION.md` or troubleshooting section!


