from pathlib import Path
import re
import orjson
from promptflow.tracing import trace
import pandas as pd
from promptflow.core import Prompty, AzureOpenAIModelConfiguration
from pf_utils import calculate_linear_probabilities, calculate_perplexity, linear_probability_to_score, perplexity_to_score
from settings import app_settings
from structured_logprobs import add_logprobs


@trace
def merge_docs(comparison_key_sets: dict):

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
        "logprobs": True,
        "stream": False,
        "response_format": {
            "type": "json_object"
        }
    }

    prompty = Prompty.load(source=BASE_DIR / "comparison.prompty",
                           model={
                               "configuration": model_config,
                               "parameters": parameters,
                               "response": "all"
                           })

    completion = prompty(comparison_key_sets=comparison_key_sets)
    result = completion.choices[0].message.content
    chat_completion = add_logprobs(completion)
    log_probs = chat_completion.log_probs[0]
    linear_probs = calculate_linear_probabilities(log_probs)
    linear_scores = linear_probability_to_score(linear_probs)
    perplexity = calculate_perplexity(log_probs)
    score = perplexity_to_score(perplexity)

    return {
        "result": orjson.loads(result),
        "overall_confidence_score": score,
        "confidence": linear_scores
    }


def normalize(value):
    if isinstance(value, str):
        return re.sub(r'[^a-z0-9]', '', value.lower())
    return value


def to_list(val):
    if isinstance(val, list):
        return [normalize(v) for v in val]
    return [normalize(val)]


def compare_json_documents(data: dict, fact_source='WOLOC') -> pd.DataFrame:
    all_docs = set()
    all_keys = data.keys()

    for key in all_keys:
        subdocs = data[key]
        all_docs.update(subdocs.keys())

    all_docs = sorted(all_docs)
    columns = ['keys'] + all_docs + ['invalidSources', 'isMatched']
    result_rows = []

    for key in all_keys:
        row = [key]
        normalized = {doc: to_list(val) for doc, val in data[key].items()}
        fact_value = normalized.get(fact_source)

        invalid_sources = []
        is_matched = True

        for doc in all_docs:
            val = data[key].get(doc)
            if isinstance(val, list):
                val_str = ', '.join(val)
            elif val is None:
                val_str = ''
            else:
                val_str = str(val)
            row.append(val_str)

            # Check mismatch
            if doc != fact_source and doc in normalized and fact_value is not None:
                if normalized[doc] != fact_value:
                    invalid_sources.append(doc)
                    is_matched = False

        row.append(', '.join(invalid_sources))
        row.append(is_matched)
        result_rows.append(row)

    df = pd.DataFrame(result_rows, columns=columns)
    return df


@trace
def compare_documents(comparison_key_sets: dict):
    merged_payload = merge_docs(comparison_key_sets)
    print(merged_payload)
    comparison_result = compare_json_documents(merged_payload['result'])
    #return pandas df to json dict
    return {
        "result": comparison_result.to_dict(orient="records"),
        "overall_confidence_score": merged_payload['overall_confidence_score'],
        "confidence": merged_payload['confidence']
    }
