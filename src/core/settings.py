import os
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV_PATH = os.environ.get(
    "DOTENV_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


class _LoggerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=DOTENV_PATH,
                                      env_file_encoding="utf-8",
                                      env_prefix="LOGGER_",
                                      extra="ignore",
                                      env_ignore_empty=True)
    log_file: Optional[str] = 'app.log'
    log_level: Optional[str] = 'INFO'
    llm_log_path: Optional[str] = 'llm.txt'


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
    embedding_model: Optional[str]
    embedding_version: Optional[str]
    embedding_base: Optional[str]
    embedding_api_key: Optional[str]


class AzureDocumentIntelligenceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,
        env_file_encoding="utf-8",
        env_prefix="AZURE_DOCUMENT_INTELLIGENCE_",
        extra="ignore",
        env_ignore_empty=True)
    api_key: str
    endpoint: str


class _RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=DOTENV_PATH,
                                      env_file_encoding="utf-8",
                                      env_prefix="REDIS_",
                                      extra="ignore",
                                      env_ignore_empty=True)

    url: str
    ttl: int


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=DOTENV_PATH,
                                      env_file_encoding="utf-8",
                                      env_prefix="OLLAMA_",
                                      extra="ignore",
                                      env_ignore_empty=True)
    model: str
    temperature: Optional[float] = 0
    base_url: str
    api_key: str


class OAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=DOTENV_PATH,
                                      env_file_encoding="utf-8",
                                      env_prefix="OPENAI_",
                                      extra="ignore",
                                      env_ignore_empty=True)
    model: str
    temperature: Optional[float] = 0
    base_url: str
    api_key: str


class MistralSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=DOTENV_PATH,
                                      env_file_encoding="utf-8",
                                      env_prefix="MISTRAL_",
                                      extra="ignore",
                                      env_ignore_empty=True)
    model: str
    temperature: Optional[float] = 0
    api_key: str


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=DOTENV_PATH,
                                      env_file_encoding="utf-8",
                                      env_prefix="EVAL_",
                                      extra="ignore",
                                      env_ignore_empty=True)
    relevancy_threshold: float


class _BaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=DOTENV_PATH,
                                      extra="ignore",
                                      arbitrary_types_allowed=True,
                                      env_ignore_empty=True)
    sanitize_answer: bool = False


class _AppSettings(BaseModel):
    logger: _LoggerSettings = _LoggerSettings()
    azure_openai: AzureOpenAISettings = AzureOpenAISettings()
    oai: OAISettings = OAISettings()
    redis: _RedisSettings = _RedisSettings()
    ollama: OllamaSettings = OllamaSettings()
    azure_document_intelligence: AzureDocumentIntelligenceSettings = AzureDocumentIntelligenceSettings(
    )
    mistral: MistralSettings = MistralSettings()


app_settings = _AppSettings()
