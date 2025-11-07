"""
Configuration module for MLflow Features Demo
Loads environment variables and provides configuration settings
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for MLflow settings"""
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
    
    # Remote MLflow Server (optional)
    MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
    MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    # LLM API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Default LLM Provider (can be 'openai' or 'gemini')
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
    
    # Model Configuration
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "150"))
    DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if cls.DEFAULT_LLM_PROVIDER == "gemini":
            if not cls.GEMINI_API_KEY:
                print("⚠️  Warning: GEMINI_API_KEY not set. Some features may not work.")
                return False
        elif cls.DEFAULT_LLM_PROVIDER == "openai":
            if not cls.OPENAI_API_KEY:
                print("⚠️  Warning: OPENAI_API_KEY not set. Some features may not work.")
                return False
        
        return True
    
    @classmethod
    def setup_mlflow(cls):
        """Setup MLflow with configuration"""
        import mlflow
        
        # Set tracking URI
        mlflow.set_tracking_uri(cls.MLFLOW_TRACKING_URI)
        
        # Set authentication if provided
        if cls.MLFLOW_TRACKING_USERNAME and cls.MLFLOW_TRACKING_PASSWORD:
            os.environ["MLFLOW_TRACKING_USERNAME"] = cls.MLFLOW_TRACKING_USERNAME
            os.environ["MLFLOW_TRACKING_PASSWORD"] = cls.MLFLOW_TRACKING_PASSWORD
        
        # Set or create experiment
        mlflow.set_experiment(cls.MLFLOW_EXPERIMENT_NAME)
        
        print(f"✅ MLflow configured:")
        print(f"   Tracking URI: {cls.MLFLOW_TRACKING_URI}")
        print(f"   Experiment: {cls.MLFLOW_EXPERIMENT_NAME}")
        
        return mlflow.get_experiment_by_name(cls.MLFLOW_EXPERIMENT_NAME)

