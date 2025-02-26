import json
import os
from pathlib import Path
import jinja2
from promptflow.core import tool
from core.settings import app_settings
from llms.ollama_llm import OllamaLLM
from tools.parsers import JSONSchemaParser
from core.logger import logger


@tool
async def extract_schema(ocr_output: str,
                         use_schema: bool,
                         json_schema: str = None):
    if (use_schema is True):
        logger.info("Using provided schema")
        logger.info("Schema: %s" % json_schema)
        return json_schema
    llm = OllamaLLM(app_settings.ollama,
                    log_file_path=app_settings.logger.llm_log_path)
    root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent
    prompt_template_path = os.path.join(
        root_path,
        "prompts",
    )
    json_schema_parser = JSONSchemaParser()

    def parse_with_retry(schema):
        return json_schema_parser.parse(schema)

    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(prompt_template_path))
    template = environment.get_template("json_schema_extraction.jinja")
    prompt = template.render(ocr_output=ocr_output)
    j_schema = llm.generate_json(prompt={
        "system": "You are an intelligent extractor",
        "user": prompt
    },
                                 parser=parse_with_retry)
    return j_schema
