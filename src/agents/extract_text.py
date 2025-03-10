import json
import os
from pathlib import Path
from typing import List
import jinja2
from promptflow.core import tool
from pydantic import BaseModel

from llms.azure_llm import AzureLLM
from llms.ollama_llm import OllamaLLM
from core.settings import app_settings
from langchain_community.cache import RedisCache
from langchain.evaluation.parsing.json_schema import JsonSchemaEvaluator
from redis import Redis
from structured_logprobs.main import add_logprobs, add_logprobs_inline
from openai.types.chat.chat_completion import Choice, ChatCompletionMessage, ChatCompletion

from tools.utils import calculate_linear_probabilities, calculate_perplexity, linear_probability_to_score, perplexity_to_score


@tool
async def extract_text(ocr_output: str, output_format: str, schema: str):
    # rd = Redis.from_url(app_settings.redis.url)
    # cache = RedisCache(redis_=rd)
    # llm = OllamaLLM(app_settings.ollama)
    llm = AzureLLM(app_settings.azure_openai)
    root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent
    prompt_template_path = os.path.join(
        root_path,
        "prompts",
    )
    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(prompt_template_path))
    template = environment.get_template("extract_json.jinja")
    prompt = template.render(json_schema=schema, ocr_output=ocr_output)
    response = llm.generate_structured_schema(
        method="json_schema",
        prompt={"user": prompt},
        json_schema=json.loads(schema),
        include_raw=True,
        logprobs=True,
    )
    raw = response["raw"]
    payload = ChatCompletion(
        id="",
        created=0,
        object="chat.completion",
        model=raw.response_metadata["model_name"],
        choices=[
            Choice(message=ChatCompletionMessage(content=raw.content,
                                                 role="assistant"),
                   index=0,
                   finish_reason="stop",
                   logprobs=raw.response_metadata["logprobs"])
        ])
    # print("RAWA", response["raw"].response_metadata)
    chat_completion = add_logprobs(payload)
    log_probs = chat_completion.log_probs[0]
    linear_probs = calculate_linear_probabilities(log_probs)
    linear_scores = linear_probability_to_score(linear_probs)
    perplexity = calculate_perplexity(log_probs)
    score = perplexity_to_score(perplexity)
    result = response["parsed"]
    if output_format == "json":
        evaluator = JsonSchemaEvaluator()
        eval_result = evaluator.evaluate_strings(reference=schema,
                                                 prediction=json.dumps(result))
        assert eval_result["score"] is not None
        if eval_result["score"] is False:
            return {
                **result, "warning": "Schema validation failed",
                "overall_confidence_score": score,
                "confidence": linear_scores
            }
        return {
            **result, "error": None,
            "overall_confidence_score": score,
            "confidence": linear_scores
        }
