import os
from pathlib import Path
from typing import List
import jinja2
from langchain_community.cache import RedisCache
from promptflow.core import tool
from redis import Redis

from llms.azure_llm import AzureLLM
from llms.ollama_llm import OllamaLLM
from core.settings import app_settings
from pydantic import BaseModel, Field, ValidationInfo, create_model, model_validator
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature, DocumentContentFormat


class Tag(BaseModel):
    id: int
    name: str
    confidence: float = Field(
        ge=0,
        le=1,
        description="The confidence of the prediction, 0 is low, 1 is high",
    )

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
    # texts: List[str]
    predictions: Tag = Field(description="Only select the most likely tags")


@tool
async def classify_document(folder_path: str, ocr_strategy: str,
                            available_tags: str):

    # raw_tags = [x.strip() for x in available_tags.split(",")]
    # tag_with_instructions = [
    #     TagWithInstructions(id=1,
    #                         name="Bankruptcy Search",
    #                         instructions="Malaysia Department of Insolvency"),
    #     TagWithInstructions(id=2,
    #                         name="Developer Letter of Understanding",
    #                         instructions="Developer Letter of Understanding")
    # ]
    # tags = [Tag(id=i, name=tag) for i, tag in enumerate(raw_tags)]
    # allowed_tags = [(tag.id, tag.name) for tag in tags]
    # allowed_tags_str = ", ".join([f"`{tag}`" for tag in allowed_tags])
    #create tags model dynamically, includes id, name and confidence score
    # TagModel = create_model(
    #     "Tags",
    #     **{tag: (float, Field(alias=tag))
    #        for tag in raw_tags},
    #     __base__=BaseModel,
    # )

    for f in os.listdir(folder_path):
        # we only need the first file (image) for classification
        with open(f"{folder_path}/{f}", "rb") as f:
            if ocr_strategy == "vision_llm":
                # rd = Redis.from_url(app_settings.redis.url)
                # cache = RedisCache(redis_=rd)
                # llm = OllamaLLM(
                #     app_settings.ollama,
                #     # cache=cache,
                #     model="llama3.2-vision:11b")
                client = DocumentIntelligenceClient(
                    endpoint=app_settings.azure_document_intelligence.endpoint,
                    credential=AzureKeyCredential(
                        app_settings.azure_document_intelligence.api_key))
                poller = client.begin_analyze_document(
                    "prebuilt-read",
                    # "prebuilt-check.us",
                    body=f.read(),
                    features=[
                        # DocumentAnalysisFeature.KEY_VALUE_PAIRS,
                        # DocumentAnalysisFeature.OCR_HIGH_RESOLUTION
                    ],
                    output_content_format=DocumentContentFormat.TEXT,
                )
                result = poller.result()

                llm = AzureLLM(app_settings.azure_openai)
                root_path = Path(os.path.dirname(
                    os.path.abspath(__file__))).parent
                prompt_template_path = os.path.join(
                    root_path,
                    "prompts",
                )
                environment = jinja2.Environment(
                    loader=jinja2.FileSystemLoader(prompt_template_path))
                template = environment.get_template("classify.jinja")
                prompt = template.render(text=result,
                                         available_tags=available_tags)
                result = llm.generate_structured_model(
                    # params={"detail": "high"},
                    # image_bytes=f.read(),
                    prompt={
                        "system": "You are a world-class text tagging system.",
                        "user": prompt,
                    },
                    response_model=Tag,
                    # validation_context={"tags": tag_with_instructions},
                    temperature=0)
            return result
