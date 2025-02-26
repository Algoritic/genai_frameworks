import json
import os
from pathlib import Path
from core.settings import AzureOpenAISettings, OllamaSettings


#compose JSON file for deepeval with extension ".deepeval"
def use_local_deepeval(config: OllamaSettings, model=None):
    root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    deepeval_config = {
        "LOCAL_MODEL_NAME": model if model else config.model,
        "LOCAL_MODEL_BASE_URL": config.base_url,
        "LOCAL_MODEL_API_KEY": config.api_key,
        "LOCAL_MODEL_FORMAT": "json",
        "USE_LOCAL_MODEL": "YES",
        "USE_AZURE_OPENAI": "NO"
    }
    deepeval_config_path = os.path.join(root_path, ".deepeval")
    with open(deepeval_config_path, "w") as f:
        json.dump(deepeval_config, f)


#compose JSON file for deepeval with extension ".deepeval"
def use_azure_deepeval(config: AzureOpenAISettings):
    root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    deepeval_config = {
        "AZURE_OPENAI_API_KEY": config.api_key,
        "AZURE_OPENAI_ENDPOINT": config.base,
        "OPENAI_API_VERSION": config.version,
        "AZURE_DEPLOYMENT_NAME": config.model_deployment,
        "USE_AZURE_OPENAI": "YES",
        "USE_LOCAL_MODEL": "NO"
    }
    deepeval_config_path = os.path.join(root_path, ".deepeval")
    with open(deepeval_config_path, "w") as f:
        json.dump(deepeval_config, f)
