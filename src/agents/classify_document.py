import os
from pathlib import Path
from typing import List
import jinja2
from langchain_community.cache import RedisCache
from promptflow.core import tool
from redis import Redis

from llms.ollama_llm import OllamaLLM
from core.settings import app_settings
from pydantic import BaseModel, ValidationInfo, model_validator


class Tag(BaseModel):
    id: int
    name: str

    @model_validator(mode="after")
    def validate_ids(self, info: ValidationInfo):
        context = info.context
        if context:
            tags: List[Tag] = context.get("tags")
            assert self.id in {tag.id
                               for tag in tags
                               }, f"Tag ID {self.id} not found in context"
            assert self.name in {
                tag.name
                for tag in tags
            }, f"Tag name {self.name} not found in context"
        return self


class TagWithInstructions(Tag):
    instructions: str


class TagRequest(BaseModel):
    texts: List[str]
    tags: List[TagWithInstructions]


class TagResponse(BaseModel):
    texts: List[str]
    predictions: List[Tag]


@tool
async def classify_document(folder_path: str, ocr_strategy: str,
                            available_tags: str):
    tags = [
        Tag(id=i + 1, name=t.strip())
        for i, t in enumerate(available_tags.split(","))
    ]
    for f in os.listdir(folder_path):
        # we only need the first file (image) for classification
        with open(f"{folder_path}/{f}", "rb") as f:
            if ocr_strategy == "vision_llm":
                rd = Redis.from_url(app_settings.redis.url)
                cache = RedisCache(redis_=rd)
                llm = OllamaLLM(app_settings.ollama,
                                cache=cache,
                                model="llama3.2-vision:11b")
                root_path = Path(os.path.dirname(
                    os.path.abspath(__file__))).parent
                prompt_template_path = os.path.join(
                    root_path,
                    "prompts",
                )
                environment = jinja2.Environment(
                    loader=jinja2.FileSystemLoader(prompt_template_path))
                template = environment.get_template("classify.jinja")
                prompt = template.render(available_tags=available_tags)
                result = llm.generate_structured_model_from_image(
                    image_bytes=f.read(),
                    prompt={
                        "system": "You are a world-class text tagging system.",
                        "user": prompt,
                    },
                    response_model=Tag,
                    validation_context={"tags": tags},
                    temperature=0)
            return result
