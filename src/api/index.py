from functools import lru_cache
from typing import Union

from langchain_redis import RedisCache
from src.api.routes import document_parser, document_classifier
from src.core.settings import app_settings

from fastapi import FastAPI
from langchain_core.globals import set_llm_cache


@lru_cache
def get_settings():
    return app_settings


def create_app():
    app = FastAPI()
    app.include_router(document_parser.document_parser_router)
    app.include_router(document_classifier.document_classifier_router)

    return app


# redis_cache = RedisCache(redis_url=app_settings.redis.url)
# set_llm_cache(redis_cache)
app = create_app()
