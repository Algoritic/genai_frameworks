import json
import math
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


def flatten_log_probs(log_probs):
    """
    Recursively extracts all log probability values from a nested dictionary.
    Ignores None values.
    """
    values = []

    if isinstance(log_probs, dict):
        for v in log_probs.values():
            values.extend(flatten_log_probs(v))
    elif isinstance(log_probs, (int, float)) and log_probs is not None:
        values.append(log_probs)

    return values


def calculate_perplexity(log_probs):
    """
    Computes perplexity given a dictionary of log probabilities.
    """
    log_prob_values = flatten_log_probs(log_probs)

    if not log_prob_values:
        return float('inf')  # Return infinity if there are no valid log probs

    avg_log_prob = sum(log_prob_values) / len(log_prob_values)
    perplexity = math.exp(-avg_log_prob)

    return perplexity


def perplexity_to_score(perplexity):
    """
    Converts perplexity to a layman-friendly score (0-100).
    Lower perplexity gets a higher score.
    """
    if perplexity <= 1:
        return 100  # Perfect confidence
    elif perplexity >= 10:
        return 0  # High uncertainty

    # Scale between 1 and 10 logarithmically
    score = max(
        0, min(100, 100 * (1 - (math.log10(perplexity) / math.log10(10)))))
    return round(score)


def calculate_linear_probabilities(log_probs):
    """
    Converts log probabilities to linear probabilities for each key.
    """

    def recursive_conversion(data):
        if isinstance(data, dict):
            return {k: recursive_conversion(v) for k, v in data.items()}
        elif isinstance(data, (int, float)) and data is not None:
            return math.exp(
                data)  # Convert log probability to linear probability
        return None

    return recursive_conversion(log_probs)


def linear_probability_to_score(linear_probs):
    """
    Converts linear probabilities to a layman-friendly score (0-100).
    """

    def recursive_conversion(data):
        if isinstance(data, dict):
            return {k: recursive_conversion(v) for k, v in data.items()}
        elif isinstance(data, (int, float)) and data is not None:
            return round(data * 100)  # Scale probability to 0-100
        return None

    return recursive_conversion(linear_probs)
