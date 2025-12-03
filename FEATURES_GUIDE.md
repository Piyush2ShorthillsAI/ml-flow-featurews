# Features Guide - Use Cases & Real-World Applications

## 📖 Overview

This guide provides **real-world use cases** for each MLflow GenAI feature, explaining **when to use them**, **why they're useful**, and **how they solve business problems**.

---

## 🎯 Feature Matrix

| Feature | Business Problem | Solution | When to Use |
|---------|-----------------|----------|-------------|
| **Prompt Registry** | Prompts scattered in code | Version control for prompts | Always |
| **Prompt Evaluation** | Don't know which prompt is better | Automated quality measurement | Before deployment |
| **Model Registry** | Multiple model versions chaos | Centralized model catalog | Always |
| **Model Evaluation** | Can't measure model quality | Automated testing framework | Before promotion |
| **A/B Testing** | Unsure which model to deploy | Production testing | Major changes |
| **Performance Monitoring** | Model degrades over time | Drift detection & alerting | Production systems |
| **Tracing** | Can't debug LLM calls | Full observability | Debugging & optimization |

---

## 💼 Use Case 1: Customer Support Chatbot

### Business Context
You're building an AI-powered customer support chatbot that handles 10,000 queries per day.

### Challenges
1. Need different responses for different query types (technical vs billing)
2. Must maintain consistent quality
3. Want to improve responses over time
4. Need to minimize API costs

### MLflow Solution

#### Step 1: Prompt Management
```python
# Register prompts for different scenarios
mlflow.genai.register_prompt(
    name="support_technical",
    template="You are a technical support specialist...",
    tags={"category": "technical", "tone": "professional"}
)

mlflow.genai.register_prompt(
    name="support_billing",
    template="You are a friendly billing assistant...",
    tags={"category": "billing", "tone": "friendly"}
)

# Set champion versions
mlflow.genai.set_prompt_alias("support_technical", "champion", version=2)
mlflow.genai.set_prompt_alias("support_billing", "champion", version=1)
```

**Business Value**:
- ✅ Organized prompts by category
- ✅ Easy to test new versions
- ✅ Quick rollback if something breaks
- ✅ Team can iterate independently

#### Step 2: Model Registration
```python
# Register model variants
# Precise model for technical queries (accurate)
precise_model = GeminiModel(temperature=0.2)
mlflow.pyfunc.log_model(..., metadata={"use_case": "technical"})

# Friendly model for billing (empathetic)
friendly_model = GeminiModel(temperature=0.7)
mlflow.pyfunc.log_model(..., metadata={"use_case": "billing"})
```

**Business Value**:
- ✅ Right model for right query type
- ✅ Optimize for quality AND cost
- ✅ Easy to switch models per category

#### Step 3: Evaluation
```python
# Create test dataset
tech_support_dataset = pd.DataFrame({
    "question": [
        "My app crashed, error code 502",
        "Can't connect to database",
        # ... 100 more technical questions
    ],
    "expected_solution": [...]
})

# Define custom scorers
def solution_accuracy_scorer(predictions, targets):
    # Check if response mentions correct solution
    scores = []
    for pred, target in zip(predictions, targets):
        if key_terms_match(pred, target):
            scores.append(1.0)
        else:
            scores.append(0.0)
    return {"solution_accuracy": np.mean(scores)}

# Evaluate
results = mlflow.evaluate(
    model="models:/support_technical/Staging",
    data=tech_support_dataset,
    evaluators=[
        solution_accuracy_scorer,
        customer_satisfaction_judge  # LLM judge
    ]
)
```

**Business Value**:
- ✅ Quantify quality before deployment
- ✅ Catch issues early
- ✅ Data-driven deployment decisions
- ✅ Prevent customer dissatisfaction

#### Step 4: A/B Testing
```python
# Test new model with 10% of traffic
class SupportBot:
    def route_query(self, user_id, query):
        # 90% get production model
        # 10% get staging model (new version)
        if hash(user_id) % 10 == 0:
            model = load_model("models:/support_technical/Staging")
            version = "staging"
        else:
            model = load_model("models:/support_technical/Production")
            version = "production"
        
        # Track metrics
        response = model.predict(query)
        mlflow.log_metric(f"{version}_latency", latency)
        mlflow.log_metric(f"{version}_satisfaction", satisfaction_score)
        
        return response

# After 1 week, analyze results
# If staging performs 10% better → promote to production
```

**Business Value**:
- ✅ Safe rollout of changes
- ✅ Real-world performance data
- ✅ Minimize risk of bad deployments
- ✅ A/B test proves value to stakeholders

#### Step 5: Monitoring
```python
# Track daily metrics
for day in range(30):
    daily_metrics = {
        "avg_latency": 450ms,
        "avg_satisfaction": 4.2/5.0,
        "total_cost": $15.30,
        "error_rate": 0.02
    }
    
    mlflow.log_metrics(daily_metrics, step=day)
    
    # Alert if metrics degrade
    if daily_metrics["avg_satisfaction"] < 3.8:
        send_alert("Customer satisfaction dropped!")
```

**Business Value**:
- ✅ Catch issues before customers complain
- ✅ Track cost trends
- ✅ Identify performance degradation
- ✅ Data for optimization decisions

### ROI Summary
- **Before**: Ad-hoc prompts, manual testing, unknown quality
- **After**: Systematic management, automated testing, data-driven decisions
- **Time Saved**: 10 hours/week on testing
- **Cost Reduction**: 20% (better model selection)
- **Quality Improvement**: 15% higher customer satisfaction

---

## 💼 Use Case 2: Content Generation Platform

### Business Context
You run a content creation platform that generates blog posts, social media content, and marketing copy for clients.

### Challenges
1. Different clients want different tones/styles
2. Need to maintain consistent brand voice per client
3. Must deliver high-quality, original content
4. Writers need to review and edit generated content

### MLflow Solution

#### Step 1: Per-Client Prompt Library
```python
# Register prompts for each client
clients = ["techstartup_xyz", "fashion_brand_abc", "finance_corp_123"]

for client in clients:
    # Blog post prompts
    mlflow.genai.register_prompt(
        name=f"{client}_blog_post",
        template=f"Write a blog post in {client_style}...",
        tags={"client": client, "content_type": "blog"}
    )
    
    # Social media prompts
    mlflow.genai.register_prompt(
        name=f"{client}_social_media",
        template=f"Write social media post for {client}...",
        tags={"client": client, "content_type": "social"}
    )

# Each client gets their own champion version
mlflow.genai.set_prompt_alias(
    f"{client}_blog_post",
    alias="champion",
    version=best_version_for_client
)
```

**Business Value**:
- ✅ Personalized content per client
- ✅ Easy to onboard new clients (register new prompts)
- ✅ Update client preferences without code changes
- ✅ A/B test different styles per client

#### Step 2: Model Variants for Different Content Types
```python
# Creative model for blog posts (engaging, varied)
creative_model = GeminiModel(
    temperature=1.0,
    max_tokens=2048
)
mlflow.pyfunc.log_model(
    ...,
    registered_model_name="content_generator",
    metadata={"use_case": "blog_posts", "variant": "creative"}
)

# Precise model for technical documentation (accurate, consistent)
precise_model = GeminiModel(
    temperature=0.2,
    max_tokens=1024
)
mlflow.pyfunc.log_model(
    ...,
    registered_model_name="content_generator",
    metadata={"use_case": "documentation", "variant": "precise"}
)

# Concise model for social media (short, punchy)
concise_model = GeminiModel(
    temperature=0.7,
    max_tokens=280  # Twitter-length
)
mlflow.pyfunc.log_model(
    ...,
    registered_model_name="content_generator",
    metadata={"use_case": "social_media", "variant": "concise"}
)
```

**Business Value**:
- ✅ Right model for right content type
- ✅ Optimize quality per use case
- ✅ Cost optimization (shorter tokens for social)

#### Step 3: Quality Evaluation
```python
# Custom evaluators for content quality
def seo_score_evaluator(predictions, targets):
    """Check SEO best practices"""
    scores = []
    for content in predictions:
        score = 0.0
        # Check keyword density
        if keyword_density(content) in [1.5, 3.0]:
            score += 0.3
        # Check meta description
        if has_meta_description(content):
            score += 0.2
        # Check heading structure
        if proper_heading_structure(content):
            score += 0.3
        # Check readability
        if readability_score(content) > 60:
            score += 0.2
        scores.append(score)
    return {"seo_score": np.mean(scores)}

def brand_voice_judge(predictions, client_id):
    """Use LLM to check brand voice alignment"""
    judge = mlflow.evaluate.make_judge(
        name="brand_voice_judge",
        judge_model="gemini:/gemini-1.5-flash",
        guidelines=f"""
        Rate how well the content matches {client_id}'s brand voice (1-5):
        5 = Perfect alignment with brand guidelines
        3 = Acceptable but needs minor adjustments
        1 = Completely off-brand
        
        Brand guidelines: {get_brand_guidelines(client_id)}
        """
    )
    return judge

# Evaluate blog posts
results = mlflow.evaluate(
    model="models:/content_generator/Staging",
    data=sample_blog_posts,
    evaluators=[
        seo_score_evaluator,
        brand_voice_judge,
        originality_checker,  # Check for plagiarism
        readability_scorer
    ]
)
```

**Business Value**:
- ✅ Automated quality checks
- ✅ Consistent brand voice
- ✅ SEO optimization
- ✅ Reduce human review time

#### Step 4: Writer Feedback Loop
```python
# Writers rate generated content
@mlflow.trace
def generate_and_review(prompt, client):
    # Generate content
    content = model.predict(prompt)
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("client", client)
        mlflow.log_param("content_type", "blog_post")
        mlflow.log_text(content, "generated_content.txt")
        
        # Writer reviews
        writer_rating = get_writer_rating(content)  # 1-5 stars
        edits_needed = count_edits(content, final_version)
        
        mlflow.log_metric("writer_rating", writer_rating)
        mlflow.log_metric("edits_needed", edits_needed)
        mlflow.log_metric("time_to_publish", time_to_publish)
    
    return content

# Analyze feedback
# If writer_rating < 3.5 → iterate on prompts
# If edits_needed > 50 → adjust model temperature
```

**Business Value**:
- ✅ Continuous improvement from writer feedback
- ✅ Reduce editing time
- ✅ Higher quality content faster
- ✅ Data-driven prompt optimization

### ROI Summary
- **Before**: Manual content creation, inconsistent quality, 4 hours per blog post
- **After**: AI-assisted generation, quality checks, 30 minutes per blog post
- **Time Saved**: 87.5% reduction in content creation time
- **Cost Reduction**: $150/post → $20/post (87% savings)
- **Quality**: Maintained or improved (via evaluation)
- **Client Satisfaction**: +25% (faster delivery, consistent brand voice)

---

## 💼 Use Case 3: Code Documentation Generator

### Business Context
Your engineering team maintains 50+ microservices and needs to keep documentation up-to-date.

### Challenges
1. Engineers don't have time to write docs
2. Documentation gets out of date quickly
3. Need consistent format across services
4. Must be technically accurate

### MLflow Solution

#### Step 1: Documentation Prompt Templates
```python
# API endpoint documentation
mlflow.genai.register_prompt(
    name="api_endpoint_docs",
    template="""
    Generate API documentation for this endpoint:
    
    Code:
    {{code}}
    
    Include:
    - Endpoint URL and method
    - Request parameters
    - Response format
    - Example request/response
    - Error codes
    
    Format: Markdown
    """,
    tags={"doc_type": "api", "format": "markdown"}
)

# Function documentation
mlflow.genai.register_prompt(
    name="function_docs",
    template="""
    Generate docstring for this function:
    
    {{code}}
    
    Include:
    - Purpose
    - Parameters (with types)
    - Return value
    - Exceptions
    - Example usage
    
    Format: {docstring_format}
    """,
    tags={"doc_type": "function", "language": "python"}
)
```

#### Step 2: Precise Model for Technical Accuracy
```python
# Use precise variant (low temperature)
docs_model = GeminiModel(
    temperature=0.2,  # Deterministic, accurate
    top_p=0.8,
    max_tokens=512
)

mlflow.pyfunc.log_model(
    python_model=docs_model,
    registered_model_name="docs_generator",
    metadata={
        "use_case": "technical_documentation",
        "quality_focus": "accuracy"
    }
)
```

#### Step 3: Accuracy Evaluation
```python
# Code-based evaluators for technical correctness
def parameter_completeness_scorer(predictions, code_snippets):
    """Check if all parameters are documented"""
    scores = []
    for doc, code in zip(predictions, code_snippets):
        params_in_code = extract_parameters(code)
        params_in_doc = extract_documented_parameters(doc)
        
        completeness = len(params_in_doc) / len(params_in_code)
        scores.append(completeness)
    
    return {"parameter_completeness": np.mean(scores)}

def type_accuracy_scorer(predictions, code_snippets):
    """Check if documented types match code"""
    scores = []
    for doc, code in zip(predictions, code_snippets):
        code_types = extract_type_hints(code)
        doc_types = extract_documented_types(doc)
        
        matches = sum([c == d for c, d in zip(code_types, doc_types)])
        accuracy = matches / len(code_types) if code_types else 0
        scores.append(accuracy)
    
    return {"type_accuracy": np.mean(scores)}

def example_validity_scorer(predictions, code_snippets):
    """Check if code examples are valid"""
    scores = []
    for doc in predictions:
        code_examples = extract_code_examples(doc)
        valid_count = 0
        for example in code_examples:
            if syntax_is_valid(example):
                valid_count += 1
        score = valid_count / len(code_examples) if code_examples else 0
        scores.append(score)
    
    return {"example_validity": np.mean(scores)}

# Evaluate
results = mlflow.evaluate(
    model="models:/docs_generator/Staging",
    data=test_code_samples,
    evaluators=[
        parameter_completeness_scorer,
        type_accuracy_scorer,
        example_validity_scorer
    ]
)

# Quality gates
if results.metrics["parameter_completeness"] > 0.95 and \
   results.metrics["type_accuracy"] > 0.90:
    promote_to_production()
```

**Business Value**:
- ✅ Automated accuracy checks
- ✅ Consistent documentation format
- ✅ Prevent incorrect documentation
- ✅ Fast feedback for prompt improvements

#### Step 4: CI/CD Integration
```python
# GitHub Action / GitLab CI
def generate_docs_on_commit(changed_files):
    """Auto-generate docs when code changes"""
    
    for file_path in changed_files:
        if file_path.endswith('.py'):
            code = read_file(file_path)
            
            # Load prompt and model
            prompt = mlflow.genai.load_prompt(
                "function_docs",
                alias="champion"
            )
            model = mlflow.pyfunc.load_model(
                "models:/docs_generator/Production"
            )
            
            # Generate docs
            docs = model.predict({
                "code": code,
                "docstring_format": "numpy"
            })
            
            # Write to docs folder
            write_docs(docs, f"docs/{file_path}.md")
            
            # Log to MLflow (for monitoring)
            with mlflow.start_run():
                mlflow.log_param("file", file_path)
                mlflow.log_metric("doc_length", len(docs))
                mlflow.log_artifact(f"docs/{file_path}.md")
    
    # Create PR with generated docs
    create_pull_request("Auto-generated documentation")
```

**Business Value**:
- ✅ Always up-to-date documentation
- ✅ Zero manual effort
- ✅ Consistent across entire codebase
- ✅ Engineers can review and approve

### ROI Summary
- **Before**: 15 minutes/function for manual docs, often skipped
- **After**: Automated generation in seconds, always done
- **Documentation Coverage**: 30% → 95%
- **Engineering Time Saved**: 20 hours/week
- **Onboarding Speed**: 50% faster (better docs)

---

## 🎯 Feature Decision Matrix

### When to Use Each Feature

#### Prompt Registry
**Use When**:
- ✅ Multiple prompts in your system
- ✅ Prompts change frequently
- ✅ Team collaborates on prompts
- ✅ Need to test prompt variants

**Skip When**:
- ❌ Single, static prompt
- ❌ Proof-of-concept only
- ❌ No collaboration needed

---

#### Model Registry
**Use When**:
- ✅ Multiple model configurations
- ✅ Need staging/production separation
- ✅ Team deploys models
- ✅ Multiple models in production

**Skip When**:
- ❌ Single model, never changes
- ❌ Personal project only
- ❌ No deployment stages

---

#### Evaluation Framework
**Use When**:
- ✅ Quality matters (always for production!)
- ✅ Testing multiple variants
- ✅ Need to justify deployment decisions
- ✅ Regulatory requirements for testing

**Skip When**:
- ❌ Experimental prototyping only
- ❌ Quality doesn't matter (rare!)

---

#### A/B Testing
**Use When**:
- ✅ Major model/prompt changes
- ✅ Production traffic available
- ✅ Need data to make decisions
- ✅ Risk of quality regression

**Skip When**:
- ❌ Clear quality improvement (evaluation shows 50%+ better)
- ❌ No production traffic yet
- ❌ Emergency fixes (deploy fast)

---

#### Performance Monitoring
**Use When**:
- ✅ Production system
- ✅ SLA requirements
- ✅ Cost matters
- ✅ Quality can degrade over time

**Skip When**:
- ❌ Development/staging only
- ❌ Short-lived experiments

---

#### Tracing
**Use When**:
- ✅ Debugging issues
- ✅ Optimizing performance
- ✅ Understanding costs
- ✅ Multi-step LLM workflows

**Skip When**:
- ❌ Everything works fine
- ❌ No performance concerns
- ❌ Single, simple LLM call

---

## 📊 Comparison: Before vs After MLflow

### Before MLflow

```python
# Scattered prompts in code
PROMPT = "Answer this question: {question}"  # Hardcoded!

# No versioning
if new_prompt_works_better:
    PROMPT = "New prompt..."  # Lost old version!

# Manual testing
for test in tests:
    response = call_llm(PROMPT.format(question=test))
    print(response)  # Human reviews manually

# Ad-hoc deployment
if looks_good:
    deploy()  # Hope for the best!

# No monitoring
# System breaks, no one knows until users complain
```

**Problems**:
- ❌ Can't rollback
- ❌ No quality metrics
- ❌ Manual testing (slow, inconsistent)
- ❌ No visibility into issues
- ❌ Hard to collaborate
- ❌ Scared to make changes

---

### After MLflow

```python
# Prompts in registry
prompt = mlflow.genai.load_prompt("qa_prompt", alias="champion")

# Versioning built-in
# Can always rollback to any previous version

# Automated testing
results = mlflow.evaluate(
    model="models:/qa_model/Staging",
    data=test_dataset,
    evaluators=[accuracy_scorer, quality_judge]
)

# Data-driven deployment
if results.metrics["quality"] > 0.85:
    client.transition_model_version_stage(
        name="qa_model",
        version=new_version,
        stage="Production"
    )

# Automated monitoring
# Alerts sent if quality drops or costs spike
```

**Benefits**:
- ✅ Easy rollback
- ✅ Quantified quality
- ✅ Automated testing (fast, consistent)
- ✅ Proactive issue detection
- ✅ Team collaboration
- ✅ Confident deployments

---

## 🚀 Implementation Roadmap

### Phase 1: Foundations (Week 1)
1. Setup MLflow server
2. Register first prompt
3. Register first model
4. Make first prediction

**Goal**: Get comfortable with basics

---

### Phase 2: Quality (Week 2-3)
1. Create test dataset
2. Build custom evaluators
3. Run first evaluation
4. Compare model versions

**Goal**: Establish quality baseline

---

### Phase 3: Production (Week 4-6)
1. Setup staging/production stages
2. Implement A/B testing
3. Deploy to production
4. Setup monitoring

**Goal**: Production-ready deployment

---

### Phase 4: Optimization (Ongoing)
1. Iterate on prompts based on data
2. Fine-tune models
3. Optimize costs
4. Improve quality metrics

**Goal**: Continuous improvement

---

## 📚 Summary

MLflow GenAI features enable **systematic, data-driven development** of LLM applications:

1. **Prompt Registry**: Version control for prompts
2. **Model Registry**: Lifecycle management for models
3. **Evaluation**: Automated quality measurement
4. **A/B Testing**: Safe production testing
5. **Monitoring**: Proactive issue detection
6. **Tracing**: Complete observability

**When to use**: Production systems where quality, cost, and reliability matter.

**Skip when**: Quick experiments or proof-of-concepts (but consider using even then for good habits!).

---

Ready to get started? See [QUICKSTART.md](QUICKSTART.md)!


