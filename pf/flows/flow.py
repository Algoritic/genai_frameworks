from pathlib import Path

from promptflow.core import AzureOpenAIModelConfiguration, Prompty
from promptflow.tracing import trace
from settings import app_settings

BASE_DIR = Path(__file__).absolute().parent


@trace
def chat(question: str = "What's the capital of France?") -> str:
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
        "top_k": 0,
        "seed": 42,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
        "stream": True,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "capital",
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the country",
                        },
                    },
                },
            },
        },
    }
    """Flow entry function."""
    prompty = Prompty.load(
        source=BASE_DIR / "sample-flow.prompty",
        model={
            "configuration": model_config,
            "parameters": parameters,
        },
    )
    output = prompty(question=question)
    return output


if __name__ == "__main__":
    from promptflow.tracing import start_trace

    start_trace()

    result = chat("What's the capital of France?")
    print(result)
