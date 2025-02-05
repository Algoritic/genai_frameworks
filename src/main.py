from langchain_openai import AzureOpenAIEmbeddings
from langchain_redis import RedisSemanticCache
from pydantic import BaseModel
from llms.azure_llm import AzureLLM
from core.settings import app_settings

from langchain_core.globals import set_llm_cache


class Answer(BaseModel):
    content: str
    explanation: str


embeddings = AzureOpenAIEmbeddings(
    model=app_settings.azure.embedding_model,
    api_key=app_settings.azure.embedding_api_key,
    api_version=app_settings.azure.embedding_version,
    azure_endpoint=app_settings.azure.embedding_base)
semantic_cache = RedisSemanticCache(embeddings=embeddings,
                                    redis_url="redis://localhost:6379",
                                    ttl=3600,
                                    distance_threshold=0.1)
set_llm_cache(semantic_cache)

azure_config = app_settings.azure
llm = AzureLLM(azure_config)
result = llm.generate_structured_model(Answer,
                                       {"user": "Hello! Could you solve 2+2?"})

print(result.explanation)

# sample_file = os.path.join(os.path.dirname(__file__), "test_sample",
#                            "sample01.pdf")
# with open(sample_file, "rb") as f:
#     path = pdf_to_images(f.read())
#     page_text = ""
#     batch_optimize_image(path)
#     for f in os.listdir(path):
#         with open(f"{path}/{f}", "rb") as f:
#             page_text += use_easy_ocr(f.read())
#     print(page_text)
#     shutil.rmtree(path)
