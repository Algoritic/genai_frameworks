from llms.azure_llm import AzureLLM

azure_config = app_settings.azure
llm = AzureLLM(azure_config)
result = llm.generate("Tell me a joke")

print(result)
