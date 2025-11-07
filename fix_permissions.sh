#!/bin/bash
# Automated fix for MLflow artifact permissions
# Run with: bash fix_permissions.sh

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║        MLflow Artifact Permission Fix - Automated                  ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Try to create /home/nifi/mlflow/artifacts with sudo
echo "Attempting to create /home/nifi/mlflow/artifacts..."
if sudo mkdir -p /home/nifi/mlflow/artifacts 2>/dev/null; then
    echo "✅ Directory created successfully"
    
    echo "Setting permissions..."
    if sudo chmod -R 777 /home/nifi/mlflow/ 2>/dev/null; then
        echo "✅ Permissions set successfully"
        
        echo ""
        echo "Verification:"
        ls -la /home/nifi/mlflow/
        
        echo ""
        echo "════════════════════════════════════════════════════════════════════"
        echo "✅ SUCCESS! Using /home/nifi/mlflow/artifacts"
        echo "════════════════════════════════════════════════════════════════════"
        echo ""
        echo "Now run: python quickstart_gemini.py"
        echo ""
    else
        echo "❌ Failed to set permissions"
        echo "Falling back to home directory..."
        USE_HOME=true
    fi
else
    echo "⚠️  Cannot create /home/nifi/ (no sudo access or permission denied)"
    echo "Using your home directory instead..."
    USE_HOME=true
fi

# Fallback: Use home directory
if [ "$USE_HOME" = true ]; then
    ARTIFACT_DIR="$HOME/mlflow_artifacts"
    
    echo ""
    echo "Creating artifact directory in your home: $ARTIFACT_DIR"
    mkdir -p "$ARTIFACT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ Directory created: $ARTIFACT_DIR"
        
        # Add to .env if not already there
        if ! grep -q "MLFLOW_ARTIFACT_ROOT" .env 2>/dev/null; then
            echo "" >> .env
            echo "# Artifact storage location (added automatically)" >> .env
            echo "MLFLOW_ARTIFACT_ROOT=$ARTIFACT_DIR" >> .env
            echo "✅ Added MLFLOW_ARTIFACT_ROOT to .env"
        else
            echo "ℹ️  MLFLOW_ARTIFACT_ROOT already in .env"
        fi
        
        echo ""
        echo "════════════════════════════════════════════════════════════════════"
        echo "✅ SUCCESS! Using $ARTIFACT_DIR"
        echo "════════════════════════════════════════════════════════════════════"
        echo ""
        echo "Your .env file now includes:"
        echo "MLFLOW_ARTIFACT_ROOT=$ARTIFACT_DIR"
        echo ""
        echo "Now run: python quickstart_gemini.py"
        echo ""
    else
        echo "❌ Failed to create directory in home"
        echo ""
        echo "Manual fix required:"
        echo "  1. Contact your system administrator"
        echo "  2. Or try: export MLFLOW_ARTIFACT_ROOT=/tmp/mlflow_artifacts"
        echo ""
    fi
fi

echo "════════════════════════════════════════════════════════════════════"

