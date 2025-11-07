#!/bin/bash
# MLflow Features Demo - Run All Examples
# This script helps you run examples in the correct order

set -e  # Exit on error

echo "========================================================================"
echo "MLflow Features Demo - Example Runner"
echo "========================================================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Please copy .env.example to .env and configure your API keys"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to run example with header
run_example() {
    local script=$1
    local name=$2
    
    echo ""
    echo "========================================================================"
    echo "Running: $name"
    echo "Script: $script"
    echo "========================================================================"
    echo ""
    
    python "$script"
    
    if [ $? -eq 0 ]; then
        echo "✅ $name completed successfully"
    else
        echo "❌ $name failed"
        read -p "Continue? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Main menu
echo "Select what to run:"
echo ""
echo "1) Quick Start (Recommended for first time)"
echo "2) All Examples (Full demonstration)"
echo "3) Beginner Track (Prompts + Evaluation + Tracing)"
echo "4) Intermediate Track (Models + Scorers + Datasets)"
echo "5) Advanced Track (Custom Tracing + Agents + E2E)"
echo "6) Individual Feature (Choose from menu)"
echo ""
read -p "Enter choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "Running Quick Start..."
        run_example "quickstart.py" "Quick Start Demo"
        ;;
    
    2)
        echo ""
        echo "Running All Examples (this will take a while)..."
        
        # Prompt Registry
        run_example "01_prompt_registry/register_prompts.py" "1. Register Prompts"
        run_example "01_prompt_registry/load_prompts.py" "2. Load Prompts"
        run_example "01_prompt_registry/prompt_lifecycle.py" "3. Prompt Lifecycle"
        
        # Prompt Evaluation
        run_example "02_prompt_evaluation/evaluate_single_prompt.py" "4. Evaluate Single Prompt"
        run_example "02_prompt_evaluation/compare_prompts.py" "5. Compare Prompts"
        run_example "02_prompt_evaluation/evaluate_with_judges.py" "6. Evaluate with Judges"
        
        # Model Registry
        run_example "03_model_registry/register_genai_model.py" "7. Register Models"
        run_example "03_model_registry/load_and_predict.py" "8. Load and Predict"
        run_example "03_model_registry/model_versioning.py" "9. Model Versioning"
        
        # Evaluation Framework
        run_example "04_evaluation_framework/custom_scorers.py" "10. Custom Scorers"
        run_example "04_evaluation_framework/llm_judges.py" "11. LLM Judges"
        
        # Tracing
        run_example "05_tracing/openai_tracing.py" "12. OpenAI Tracing"
        run_example "05_tracing/custom_tracing.py" "13. Custom Tracing"
        
        # Evaluation Datasets
        run_example "06_evaluation_datasets/create_from_data.py" "14. Create Datasets"
        
        # ResponsesAgent
        run_example "07_responses_agent/simple_agent.py" "15. Simple Agent"
        
        # End-to-End
        run_example "08_end_to_end/full_pipeline.py" "16. Full Pipeline"
        
        echo ""
        echo "✅ All examples completed!"
        ;;
    
    3)
        echo ""
        echo "Running Beginner Track..."
        run_example "01_prompt_registry/register_prompts.py" "Register Prompts"
        run_example "01_prompt_registry/load_prompts.py" "Load Prompts"
        run_example "02_prompt_evaluation/evaluate_single_prompt.py" "Evaluate Prompt"
        run_example "05_tracing/openai_tracing.py" "OpenAI Tracing"
        ;;
    
    4)
        echo ""
        echo "Running Intermediate Track..."
        run_example "03_model_registry/register_genai_model.py" "Register Models"
        run_example "03_model_registry/load_and_predict.py" "Load and Predict"
        run_example "04_evaluation_framework/custom_scorers.py" "Custom Scorers"
        run_example "06_evaluation_datasets/create_from_data.py" "Create Datasets"
        ;;
    
    5)
        echo ""
        echo "Running Advanced Track..."
        run_example "05_tracing/custom_tracing.py" "Custom Tracing"
        run_example "07_responses_agent/simple_agent.py" "ResponsesAgent"
        run_example "08_end_to_end/full_pipeline.py" "End-to-End Pipeline"
        ;;
    
    6)
        echo ""
        echo "Select Feature:"
        echo "1) Prompt Registry"
        echo "2) Prompt Evaluation"
        echo "3) Model Registry"
        echo "4) Evaluation Framework"
        echo "5) Tracing"
        echo "6) Evaluation Datasets"
        echo "7) ResponsesAgent"
        echo "8) End-to-End Pipeline"
        echo ""
        read -p "Enter choice (1-8): " feature_choice
        
        case $feature_choice in
            1)
                run_example "01_prompt_registry/register_prompts.py" "Register Prompts"
                ;;
            2)
                run_example "02_prompt_evaluation/evaluate_single_prompt.py" "Evaluate Prompt"
                ;;
            3)
                run_example "03_model_registry/register_genai_model.py" "Register Models"
                ;;
            4)
                run_example "04_evaluation_framework/custom_scorers.py" "Custom Scorers"
                ;;
            5)
                run_example "05_tracing/openai_tracing.py" "OpenAI Tracing"
                ;;
            6)
                run_example "06_evaluation_datasets/create_from_data.py" "Create Datasets"
                ;;
            7)
                run_example "07_responses_agent/simple_agent.py" "Simple Agent"
                ;;
            8)
                run_example "08_end_to_end/full_pipeline.py" "Full Pipeline"
                ;;
            *)
                echo "Invalid choice"
                exit 1
                ;;
        esac
        ;;
    
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "✅ Execution Complete!"
echo "========================================================================"
echo ""
echo "Next Steps:"
echo "1. View results in MLflow UI: http://localhost:5000"
echo "2. Check GETTING_STARTED.md for more examples"
echo "3. Read README.md for detailed documentation"
echo ""

