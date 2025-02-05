import json
import os
from pathlib import Path
import jinja2
from promptflow.core import tool

from llms.ollama_llm import OllamaLLM
from core.settings import app_settings
from langchain_community.cache import RedisCache
from langchain.evaluation.parsing.json_schema import JsonSchemaEvaluator
from redis import Redis


@tool
async def extract_text(ocr_output: str, output_format: str, schema: str):

    # rd = Redis.from_url(app_settings.redis.url)
    # cache = RedisCache(redis_=rd)
    llm = OllamaLLM(app_settings.ollama, )
    root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent
    prompt_template_path = os.path.join(
        root_path,
        "prompts",
    )
    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(prompt_template_path))
    template = environment.get_template("extract_json.jinja")
    prompt = template.render(json_schema=schema, ocr_output=ocr_output)
    result = await llm.agenerate_structured_schema(
        prompt={
            "user": prompt,
            "system": "You are an intelligent text extractor."
        },
        json_schema=json.loads(schema))
    if output_format == "json":
        evaluator = JsonSchemaEvaluator()
        eval_result = evaluator.evaluate_strings(reference=schema,
                                                 prediction=json.dumps(result))
        assert eval_result["score"] is not None
        if eval_result["score"] is False:
            return {**result, "warning": "Schema validation failed"}
        return {**result, "error": None}
