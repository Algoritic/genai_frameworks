from pathlib import Path
import orjson
from promptflow.core import tool

from promptflow.core import Prompty, AzureOpenAIModelConfiguration
from settings import app_settings
from logger import logger
from pf_utils import calculate_linear_probabilities, calculate_perplexity, linear_probability_to_score, perplexity_to_score
from structured_logprobs import add_logprobs
from promptflow.tracing import trace


@trace
@tool
async def structured_extraction(ocr_result: str, json_schema: dict):
    BASE_DIR = Path(__file__).absolute().parent.parent / "flows"
    model_config: AzureOpenAIModelConfiguration = {
        "type": "azure_openai",
        "azure_deployment": app_settings.azure_openai.model_deployment,
        "azure_endpoint": app_settings.azure_openai.base,
        "api_version": app_settings.azure_openai.version,
        "api_key": app_settings.azure_openai.api_key,
    }
    parameters = {
        "temperature": app_settings.azure_openai.temperature,
        "max_tokens": app_settings.azure_openai.max_tokens,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
        "stream": False,
        "logprobs": True,
        "response_format": {
            "type": "json_schema",
            "json_schema": json_schema
        }
    }
    prompty = Prompty.load(source=BASE_DIR / "extract-json.prompty",
                           model={
                               "configuration": model_config,
                               "parameters": parameters,
                               "response": "all"
                           })
    completion = prompty(json_schema=json_schema, ocr_output=ocr_result)
    content = completion.choices[0].message.content
    chat_completion = add_logprobs(completion)
    log_probs = chat_completion.log_probs[0]
    linear_probs = calculate_linear_probabilities(log_probs)
    linear_scores = linear_probability_to_score(linear_probs)
    perplexity = calculate_perplexity(log_probs)
    score = perplexity_to_score(perplexity)

    result = orjson.loads(content)

    logger.debug(f"Structured extraction result: {content}")
    return {
        "result": result,
        "overall_confidence_score": score,
        "confidence": linear_scores
    }
