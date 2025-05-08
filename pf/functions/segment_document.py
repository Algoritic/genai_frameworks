from pathlib import Path
import re
from typing import List

from openai import BaseModel
import orjson
from logger import logger
from promptflow.core import Prompty, AzureOpenAIModelConfiguration
from settings import app_settings
from pf_utils import calculate_linear_probabilities, calculate_perplexity, linear_probability_to_score, perplexity_to_score
from structured_logprobs import add_logprobs
from promptflow.tracing import trace
from collections import defaultdict


class Segment(BaseModel):
    document_name: str
    sub_document_name: str
    confidence: float
    reasoning: str


def group_tags(data):
    grouped = defaultdict(lambda: {"indices": [], "seconds": set()})

    for index, (first, second) in enumerate(data):
        grouped[first]["indices"].append(index)
        if second != first:
            grouped[first]["seconds"].add(second)

    result = []
    for key, info in grouped.items():
        indices = info["indices"]
        result.append({
            "documentName":
            key,
            "indices":
            indices,
            "range": (indices[0], indices[-1]) if len(indices) > 1 else
            (indices[0], indices[0]),
            "subDocumentNames":
            sorted(info["seconds"]) if info["seconds"] else []
        })

    return result


@trace
#pages is list of extracted text from pdf, page by page
def segment(json_schema: dict, pages=List[str]):

    logger.info(f"Segmenting {len(pages)} pages")

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

    prompty = Prompty.load(source=BASE_DIR / "classify.prompty",
                           model={
                               "configuration": model_config,
                               "parameters": parameters,
                               "response": "all"
                           })
    metrics = [None] * len(pages)
    page_tags = [None] * len(pages)
    for idx, page in enumerate(pages):

        current_page_number = idx + 1
        logger.info(
            f"Analyzing transition between page {idx} and page {current_page_number}"
        )

        prev_page_doc_name, prev_page_sub_doc_name = page_tags[
            idx - 1] if idx > 0 else (None, None)

        #fallback to previous page tag if page with the word "CC" is found
        pattern = r"(?<!\w)[cC]\s*\.?\s*[cC](?=\b|[^a-zA-Z])"
        matches = re.findall(pattern, page)
        if len(matches) > 0:
            page_tags.insert(idx, (prev_page_doc_name, prev_page_sub_doc_name))
            continue

        completion = prompty(page_content=page,
                             prev_doc_name=prev_page_doc_name,
                             prev_doc_sub_name=prev_page_sub_doc_name)
        content = completion.choices[0].message.content
        chat_completion = add_logprobs(completion)
        log_probs = chat_completion.log_probs[0]
        linear_probs = calculate_linear_probabilities(log_probs)
        linear_scores = linear_probability_to_score(linear_probs)
        perplexity = calculate_perplexity(log_probs)
        score = perplexity_to_score(perplexity)

        # result_string = ''.join(map(str, output))
        result = orjson.loads(content)
        page_tags[idx] = (result["document_name"], result["sub_document_name"])
        metrics[idx] = {
            "overall_confidence_score": score,
            "confidence": linear_scores
        }

        logger.info(f"Page {current_page_number} tags: {result}")

    grouped_tags = group_tags([(page_tags[i][0], page_tags[i][1])
                               for i in range(len(page_tags))])

    return {"tags": grouped_tags, "metrics": metrics}
