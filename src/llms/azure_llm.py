import base64
import json
import instructor
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
from core.settings import AzureSettings
from llms.llm import LLMBase
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.callbacks.tracers.logging import LoggingCallbackHandler
from langchain.output_parsers import RetryOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from core.logger import logger


class AzureLLM(LLMBase):

    def __init__(self, config: AzureSettings):
        super().__init__()
        self.parser = JsonOutputParser()
        self.config = config
        self.model = AzureChatOpenAI(
            deployment_name=config.model_deployment,
            api_key=config.api_key,
            openai_api_type="azure",
            azure_endpoint=config.base,
            api_version=config.version,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            callbacks=[LoggingCallbackHandler(logger=logger)])

    def generate(self, prompt: dict[str, str], params=None, **kwargs):
        chat_template = ChatPromptTemplate.from_messages([
            ("system", prompt.get("system", "You are a helpful assistant.")),
            ("user", "{user_input}"),
        ])
        messages = chat_template.format_messages(
            user_input=prompt.get("user", "Answer my question"))
        response = self.model.invoke(messages, **kwargs)
        return response

    async def agenerate(self, prompt: dict[str, str], params=None, **kwargs):
        chat_template = ChatPromptTemplate.from_messages([
            ("system", prompt.get("system", "You are a helpful assistant.")),
            ("user", "{user_input}"),
        ])
        messages = chat_template.format_messages(
            user_input=prompt.get("user", "Answer my question"))
        response = await self.model.ainvoke(messages)
        return response

    def generate_structured_model(self,
                                  response_model: BaseModel,
                                  prompt: dict[str, str] = None,
                                  params=None,
                                  **kwargs):
        root_client = self.model.root_client
        client = instructor.from_openai(root_client)
        result = client.chat.completions.create(messages=[{
            "role":
            "system",
            "content":
            prompt.get("system", "You are a helpful assistant.")
        }, {
            "role":
            "user",
            "content":
            prompt.get("user", "Answer my question")
        }],
                                                model=self.config.model,
                                                response_model=response_model,
                                                **kwargs)
        return result

    async def agenerate_structured_model(self,
                                         response_model: BaseModel,
                                         prompt: dict[str, str] = None,
                                         params=None,
                                         **kwargs):
        root_client = self.model.root_async_client
        client = instructor.from_openai(root_client)
        result = await client.chat.completions.create(
            messages=[{
                "role":
                "system",
                "content":
                prompt.get("system", "You are a helpful assistant.")
            }, {
                "role": "user",
                "content": prompt.get("user", "Answer my question")
            }],
            model=self.config.model,
            response_model=response_model)
        return result

    def generate_json(self, prompt: dict[str, str], params=None, **kwargs):
        chat_template = ChatPromptTemplate.from_messages([
            ("system", prompt.get("system", "You are a helpful assistant.")),
            ("user", "{user_input}"),
        ])
        completion_chain = chat_template | self.model
        retry_parser = RetryOutputParser.from_llm(parser=self.parser,
                                                  llm=self.model)

        def parse_with_retry(x):
            return retry_parser.parse_with_prompt(
                prompt_value=x["prompt_value"],
                completion=x["completion"].content)

        main_chain = RunnableParallel(
            completion=completion_chain,
            prompt_value=chat_template) | RunnableLambda(parse_with_retry)
        response = main_chain.invoke(
            {"user_input": prompt.get("user", "Answer my question")},
            config={'tags': ['ollama']},
            **kwargs)
        return json.dumps(response)

    def generate_structured_model_from_image(self,
                                             image_bytes,
                                             response_model,
                                             prompt: dict[str, str] = None,
                                             params=None,
                                             **kwargs):
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        payload = [
            {
                "role": "system",
                "content": prompt.get("system", "You are a helpful assistant.")
            },
            {
                "role":
                "user",
                "content": [{
                    "type":
                    "text",
                    "content":
                    prompt.get("user", "Tell me what you see")
                }, {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64," + image_base64,
                    "detail": params.get('detail', 'high')
                }]
            },
        ]
        root_client = self.model.root_client
        client = instructor.from_openai(root_client)
        result = client.chat.completions.create(messages=payload,
                                                model=self.config.model,
                                                response_model=response_model,
                                                **kwargs)
        return result

    async def agenerate_structured_model_from_image(self,
                                                    image_bytes,
                                                    response_model,
                                                    prompt: dict[str,
                                                                 str] = None,
                                                    params=None,
                                                    **kwargs):
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        payload = [
            {
                "role": "system",
                "content": prompt.get("system", "You are a helpful assistant.")
            },
            {
                "role":
                "user",
                "content": [{
                    "type":
                    "text",
                    "content":
                    prompt.get("user", "Tell me what you see")
                }, {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64," + image_base64,
                    "detail": params.get('detail', 'high')
                }]
            },
        ]
        root_client = self.model.root_async_client
        client = instructor.from_openai(root_client)
        result = await client.chat.completions.create(
            messages=payload,
            model=self.config.model,
            response_model=response_model,
            **kwargs)
        return result

    def generate_structured_schema(self,
                                   json_schema,
                                   prompt: dict[str, str] = None,
                                   params=None,
                                   **kwargs):
        chat_template = ChatPromptTemplate.from_messages([
            ("system", prompt.get("system", "You are a helpful assistant.")),
            ("user", "{user_input}"),
        ])
        messages = chat_template.format_messages(
            user_input=prompt.get("user", "Answer my question"))
        response = self.model.with_structured_output(json_schema).invoke(
            messages, **kwargs)
        return response

    async def agenerate_structured_schema(self,
                                          json_schema,
                                          prompt: dict[str, str] = None,
                                          params=None,
                                          **kwargs):
        chat_template = ChatPromptTemplate.from_messages([
            ("system", prompt.get("system", "You are a helpful assistant.")),
            ("user", "{user_input}"),
        ])
        messages = chat_template.format_messages(
            user_input=prompt.get("user", "Answer my question"))
        response = await self.model.with_structured_output(
            json_schema).ainvoke(messages, **kwargs)
        return response

    def generate_structured_schema_from_image(self,
                                              image_bytes,
                                              json_schema,
                                              prompt: dict[str, str] = None,
                                              params=None,
                                              **kwargs):
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        payload = [
            {
                "role": "system",
                "content": prompt.get("system", "You are a helpful assistant.")
            },
            {
                "role":
                "user",
                "content": [{
                    "type":
                    "text",
                    "content":
                    prompt.get("user", "Tell me what you see")
                }, {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64," + image_base64,
                    "detail": params.get('detail', 'high')
                }]
            },
        ]
        response = self.model.with_structured_output(json_schema).invoke(
            payload, **kwargs)
        return response

    async def agenerate_structured_schema_from_image(self,
                                                     image_bytes,
                                                     json_schema,
                                                     prompt: dict[str,
                                                                  str] = None,
                                                     params=None,
                                                     **kwargs):
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        payload = [
            {
                "role": "system",
                "content": prompt.get("system", "You are a helpful assistant.")
            },
            {
                "role":
                "user",
                "content": [{
                    "type":
                    "text",
                    "content":
                    prompt.get("user", "Tell me what you see")
                }, {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64," + image_base64,
                    "detail": params.get('detail', 'high')
                }]
            },
        ]
        response = await self.model.with_structured_output(
            json_schema).ainvoke(payload, **kwargs)
        return response
