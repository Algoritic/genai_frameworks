import os
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from logger import logger

DOTENV_PATH = os.environ.get(
    "DOTENV_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

logger.debug(f"Loading environment variables from {DOTENV_PATH}")


class AzureOpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=DOTENV_PATH,
                                      env_file_encoding="utf-8",
                                      env_prefix="AZURE_OPENAI_",
                                      extra="ignore",
                                      env_ignore_empty=True)
    base: str
    model: str
    max_tokens: int
    api_key: str
    version: str
    model_deployment: str
    temperature: Optional[float] = 0
    # embedding_model: Optional[str]
    # embedding_version: Optional[str]
    # embedding_base: Optional[str]
    # embedding_api_key: Optional[str]


class AzureDocumentIntelligenceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,
        env_file_encoding="utf-8",
        env_prefix="AZURE_DOCUMENT_INTELLIGENCE_",
        extra="ignore",
        env_ignore_empty=True)
    api_key: str
    endpoint: str


class _AppSettings(BaseModel):
    # logger: _LoggerSettings = _LoggerSettings()
    azure_openai: AzureOpenAISettings = AzureOpenAISettings()
    # oai: OAISettings = OAISettings()
    # redis: _RedisSettings = _RedisSettings()
    # ollama: OllamaSettings = OllamaSettings()
    azure_document_intelligence: AzureDocumentIntelligenceSettings = AzureDocumentIntelligenceSettings(
    )
    # mistral: MistralSettings = MistralSettings()


app_settings = _AppSettings()
