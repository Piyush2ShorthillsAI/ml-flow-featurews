# Documentation Index

## 📚 Complete Documentation Guide

Welcome to the MLflow GenAI Features documentation! This index helps you find the right documentation for your needs.

---

## 🚀 Start Here

### New to the Project?
1. **[README.md](README.md)** - Project overview and quick links
2. **[QUICKSTART.md](QUICKSTART.md)** - 15-minute hands-on tutorial
3. Run the examples in `01_prompt_registry/` and `03_model_registry/`

### Want to Understand How It Works?
1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and internals
2. **[DOCUMENTATION.md](DOCUMENTATION.md)** - Comprehensive feature guide

### Looking for Specific Features?
1. **[FEATURES_GUIDE.md](FEATURES_GUIDE.md)** - Real-world use cases
2. Feature-specific READMEs (see below)

---

## 📖 Documentation Structure

```
Documentation Hierarchy:

README.md (Start Here)
├─ QUICKSTART.md (Hands-on Tutorial)
├─ DOCUMENTATION.md (Comprehensive Guide)
├─ ARCHITECTURE.md (Technical Deep Dive)
└─ FEATURES_GUIDE.md (Use Cases)

Feature Documentation:
├─ 01_prompt_registry/README.md
├─ 03_model_registry/README.md
└─ [Other directories have inline documentation]
```

---

## 📋 Documentation by Goal

### 🎯 "I want to get started quickly"
→ **[QUICKSTART.md](QUICKSTART.md)** (15 minutes)

### 🎓 "I want to learn everything"
→ **[DOCUMENTATION.md](DOCUMENTATION.md)** (comprehensive)

### 🏗️ "I want to understand the architecture"
→ **[ARCHITECTURE.md](ARCHITECTURE.md)** (technical details)

### 💼 "I want to see real-world examples"
→ **[FEATURES_GUIDE.md](FEATURES_GUIDE.md)** (use cases)

### 🔧 "I want to work with prompts"
→ **[01_prompt_registry/README.md](01_prompt_registry/README.md)**

### 🤖 "I want to work with models"
→ **[03_model_registry/README.md](03_model_registry/README.md)**

### 🐛 "I'm stuck with an error"
→ **[QUICKSTART.md - Troubleshooting](QUICKSTART.md#common-issues--solutions)**
→ **[DOCUMENTATION.md - Troubleshooting](DOCUMENTATION.md#troubleshooting)**

---

## 📚 Full Documentation List

### Core Documentation

| Document | Length | Purpose | Audience |
|----------|--------|---------|----------|
| **[README.md](README.md)** | 5 min | Project overview, navigation | Everyone |
| **[QUICKSTART.md](QUICKSTART.md)** | 15 min | Hands-on tutorial | Beginners |
| **[DOCUMENTATION.md](DOCUMENTATION.md)** | 30 min | Complete feature guide | Intermediate |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | 20 min | System design & internals | Advanced |
| **[FEATURES_GUIDE.md](FEATURES_GUIDE.md)** | 20 min | Real-world use cases | All levels |

### Feature-Specific Documentation

| Document | Length | Purpose |
|----------|--------|---------|
| **[01_prompt_registry/README.md](01_prompt_registry/README.md)** | 15 min | Prompt management deep dive |
| **[03_model_registry/README.md](03_model_registry/README.md)** | 20 min | Model lifecycle guide |

### Configuration & Setup

| Document | Purpose |
|----------|---------|
| **[config.py](config.py)** | Configuration management (inline docs) |
| **[.env.example](.env.example)** | Environment variables template |
| **[requirements.txt](requirements.txt)** | Python dependencies |

### Additional Resources

| Document | Purpose |
|----------|---------|
| **[REMOTE_SERVER_GUIDE.md](REMOTE_SERVER_GUIDE.md)** | Remote MLflow setup |
| **[QUICK_SETUP_REMOTE.txt](QUICK_SETUP_REMOTE.txt)** | Quick remote setup |
| **[GEMINI_QUICKSTART.md](GEMINI_QUICKSTART.md)** | Gemini-specific guide |

---

## 🗺️ Learning Paths

### Path 1: Quick Learner (1 day)
1. Read [README.md](README.md) - 5 min
2. Complete [QUICKSTART.md](QUICKSTART.md) - 15 min
3. Run examples - 30 min
4. Skim [DOCUMENTATION.md](DOCUMENTATION.md) - 15 min

**Outcome**: Can use basic features

---

### Path 2: Comprehensive Learner (1 week)
**Day 1**: Setup & Basics
- Read [QUICKSTART.md](QUICKSTART.md)
- Complete tutorial
- Explore MLflow UI

**Day 2-3**: Prompts
- Read [01_prompt_registry/README.md](01_prompt_registry/README.md)
- Run all prompt examples
- Build custom prompts

**Day 4-5**: Models
- Read [03_model_registry/README.md](03_model_registry/README.md)
- Register models
- Run evaluation examples

**Day 6**: Architecture
- Read [ARCHITECTURE.md](ARCHITECTURE.md)
- Understand system design
- Review code flow

**Day 7**: Use Cases
- Read [FEATURES_GUIDE.md](FEATURES_GUIDE.md)
- Apply to your use case
- Plan implementation

**Outcome**: Expert-level understanding

---

### Path 3: Practical Builder (Ongoing)
1. Read [QUICKSTART.md](QUICKSTART.md) - Day 1
2. Build feature as you go:
   - Need prompts? → [01_prompt_registry/README.md](01_prompt_registry/README.md)
   - Need models? → [03_model_registry/README.md](03_model_registry/README.md)
   - Stuck? → Troubleshooting sections
3. Reference [ARCHITECTURE.md](ARCHITECTURE.md) when needed
4. Check [FEATURES_GUIDE.md](FEATURES_GUIDE.md) for ideas

**Outcome**: Learn by doing

---

## 📊 Documentation Coverage

### What's Documented

✅ **Setup & Configuration**
- Environment variables
- MLflow server setup
- API key configuration
- Remote server connection

✅ **Core Features**
- Prompt Registry (comprehensive)
- Model Registry (comprehensive)
- Evaluation Framework (comprehensive)
- Tracing (overview)

✅ **Workflows**
- Development workflow
- Evaluation workflow
- Deployment workflow
- Monitoring workflow

✅ **Use Cases**
- Customer support chatbot
- Content generation
- Code documentation
- 3 detailed examples

✅ **Code Explanations**
- What each file does
- How code works internally
- When to use each feature
- Best practices

✅ **Troubleshooting**
- Common issues
- Solutions
- Debugging tips

### What's Not Documented (Yet)

⏳ **Advanced Topics** (available in code, not in detailed docs):
- 02_prompt_evaluation/ (examples exist, no dedicated README)
- 04_evaluation_framework/ (examples exist, no dedicated README)
- 05_tracing/ (examples exist, no dedicated README)
- 06_evaluation_datasets/ (examples exist)
- 07_responses_agent/ (examples exist)
- 08_end_to_end/ (examples exist)

**Note**: All directories have working code with inline comments. Comprehensive READMEs prioritized for most-used features (prompts, models).

---

## 🔍 Find Information Quickly

### By Topic

**Configuration**
- Setup: [QUICKSTART.md - Setup](QUICKSTART.md#5-minute-setup)
- Environment: [.env.example](.env.example)
- API Keys: [QUICKSTART.md - Configure](QUICKSTART.md#step-3-configure-environment)

**Prompts**
- Basics: [QUICKSTART.md - Workflow 1](QUICKSTART.md#workflow-1-prompt-management-5-minutes)
- Deep dive: [01_prompt_registry/README.md](01_prompt_registry/README.md)
- Use cases: [FEATURES_GUIDE.md - Use Case 1](FEATURES_GUIDE.md#use-case-1-customer-support-chatbot)

**Models**
- Basics: [QUICKSTART.md - Workflow 2](QUICKSTART.md#workflow-2-model-registration-5-minutes)
- Deep dive: [03_model_registry/README.md](03_model_registry/README.md)
- Architecture: [ARCHITECTURE.md - Model Registry](ARCHITECTURE.md#4-model-registry-layer-03_model_registry)

**Evaluation**
- Overview: [DOCUMENTATION.md - Evaluation](DOCUMENTATION.md#feature-4-model-evaluation)
- Custom scorers: [ARCHITECTURE.md - Custom Scorers](ARCHITECTURE.md#custom-scorers)
- Use cases: [FEATURES_GUIDE.md - Evaluation](FEATURES_GUIDE.md#step-3-quality-evaluation)

**Deployment**
- Stages: [03_model_registry/README.md - Lifecycle](03_model_registry/README.md#6-model-versioning.py)
- A/B testing: [03_model_registry/README.md - A/B Testing](03_model_registry/README.md#5-model_ab_testing.py)
- Monitoring: [DOCUMENTATION.md - Monitoring](DOCUMENTATION.md#feature-7-performance-monitoring)

### By Question

**"How do I...?"**
- Get started? → [QUICKSTART.md](QUICKSTART.md)
- Register a prompt? → [01_prompt_registry/README.md - register_prompts.py](01_prompt_registry/README.md#1-register_promptspy)
- Register a model? → [03_model_registry/README.md - register_gemini_model.py](03_model_registry/README.md#1-register_gemini_modelpy-)
- Evaluate quality? → [DOCUMENTATION.md - Evaluation](DOCUMENTATION.md#feature-4-model-evaluation)
- Deploy to production? → [03_model_registry/README.md - Workflow](03_model_registry/README.md#typical-workflow)

**"What is...?"**
- A prompt registry? → [DOCUMENTATION.md - Prompt Registry](DOCUMENTATION.md#feature-1-prompt-registry)
- A model registry? → [DOCUMENTATION.md - Model Registry](DOCUMENTATION.md#feature-3-model-registry)
- PyFunc? → [ARCHITECTURE.md - PyFunc](ARCHITECTURE.md#pyfunc-model-wrapper)
- An LLM judge? → [ARCHITECTURE.md - LLM Judges](ARCHITECTURE.md#llm-judges)

**"Why should I...?"**
- Use MLflow? → [FEATURES_GUIDE.md - Comparison](FEATURES_GUIDE.md#comparison-before-vs-after-mlflow)
- Version prompts? → [FEATURES_GUIDE.md - Use Case 1](FEATURES_GUIDE.md#use-case-1-customer-support-chatbot)
- Evaluate models? → [FEATURES_GUIDE.md - Decision Matrix](FEATURES_GUIDE.md#feature-decision-matrix)

**"When should I...?"**
- Use A/B testing? → [FEATURES_GUIDE.md - A/B Testing](FEATURES_GUIDE.md#ab-testing)
- Set up monitoring? → [FEATURES_GUIDE.md - Monitoring](FEATURES_GUIDE.md#performance-monitoring)
- Use tracing? → [FEATURES_GUIDE.md - Tracing](FEATURES_GUIDE.md#tracing)

---

## 🎯 Recommended Reading Order

### For Beginners
1. [README.md](README.md) - Overview
2. [QUICKSTART.md](QUICKSTART.md) - Hands-on
3. [01_prompt_registry/README.md](01_prompt_registry/README.md) - Prompts
4. [03_model_registry/README.md](03_model_registry/README.md) - Models
5. [FEATURES_GUIDE.md](FEATURES_GUIDE.md) - Use cases

### For Intermediate Users
1. [DOCUMENTATION.md](DOCUMENTATION.md) - Complete guide
2. [ARCHITECTURE.md](ARCHITECTURE.md) - System design
3. [FEATURES_GUIDE.md](FEATURES_GUIDE.md) - Advanced use cases
4. Feature-specific READMEs as needed

### For Advanced Users
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Deep dive
2. Code files (with inline documentation)
3. [FEATURES_GUIDE.md](FEATURES_GUIDE.md) - Optimization ideas
4. MLflow official docs (links in README)

---

## 📖 Documentation Statistics

- **Total Documents**: 10+ comprehensive guides
- **Total Words**: ~50,000+ words
- **Total Code Examples**: 100+ snippets
- **Total Use Cases**: 3 detailed real-world scenarios
- **Coverage**: Core features fully documented
- **Maintenance**: Active (updated with features)

---

## 🆘 Still Can't Find What You Need?

1. **Search within files**: Use Ctrl+F in each document
2. **Check code files**: Inline documentation in Python files
3. **MLflow docs**: [MLflow Documentation](https://mlflow.org/docs/latest/)
4. **Gemini docs**: [Gemini API Documentation](https://ai.google.dev/docs)

---

## 📝 Documentation Maintenance

**Last Updated**: November 2024

**Update Frequency**: As features are added/changed

**Contributing**: Feel free to improve documentation!

---

**Happy Learning! 📚**


