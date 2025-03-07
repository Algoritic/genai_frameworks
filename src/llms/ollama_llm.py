import base64
import json
import instructor
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from callbacks.confident_callback import DeepEvalCallbackHandler
from core.logger import logger
from core.settings import OllamaSettings
from llms.llm import LLMBase
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain.callbacks.tracers.logging import LoggingCallbackHandler
from langchain.output_parsers import RetryOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.messages import HumanMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from deepeval.metrics import AnswerRelevancyMetric, ToxicityMetric, BiasMetric


class OllamaLLM(LLMBase):

    #TODO: Accept eval metrics
    def __init__(self,
                 config: OllamaSettings,
                 cache=None,
                 model=None,
                 **kwargs):

        super().__init__()

        self.parser = JsonOutputParser()
        self.config = config

        answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7,
                                                        model="gpt-4o",
                                                        include_reason=False)

        toxicity_metric = ToxicityMetric(threshold=0.7,
                                         model="gpt-4o",
                                         include_reason=False)
        bias_metric = BiasMetric(threshold=0.7,
                                 model="gpt-4o",
                                 include_reason=False)

        deepeval_callback = DeepEvalCallbackHandler(
            implementation_name=f"ollama {model if model else config.model}",
            logger=logger,
            metrics=[
                answer_relevancy_metric,
                toxicity_metric,
                bias_metric,
            ])
        self.model = ChatOllama(
            cache=cache,
            model=model if model else config.model,
            temperature=config.temperature,
            callbacks=[LoggingCallbackHandler(logger=logger)],
        )

    def generate(self, prompt: dict[str, str], params=None, **kwargs):
        chat_template = ChatPromptTemplate.from_messages([
            ("system", prompt.get("system", "You are a helpful assistant.")),
            ("user", "{user_input}"),
        ])
        messages = chat_template.format_messages(
            user_input=prompt.get("user", "Answer my question"))
        response = self.model.invoke(messages,
                                     config={'tags': ['ollama']},
                                     **kwargs)
        return response

    async def agenerate(self, prompt: dict[str, str], params=None, **kwargs):
        chat_template = ChatPromptTemplate.from_messages([
            ("system", prompt.get("system", "You are a helpful assistant.")),
            ("user", "{user_input}"),
        ])
        messages = chat_template.format_messages(
            user_input=prompt.get("user", "Answer my question"))
        response = await self.model.ainvoke(messages, **kwargs)
        return response

    def generate_structured_model(self,
                                  response_model: BaseModel,
                                  prompt: dict[str, str] = None,
                                  params=None,
                                  **kwargs):
        chat_template = ChatPromptTemplate.from_messages([
            ("system", prompt.get("system", "You are a helpful assistant.")),
            ("user", "{user_input}"),
        ])
        messages = chat_template.format_messages(
            user_input=prompt.get("user", "Answer my question"))
        response = self.model.with_structured_output(
            response_model.model_json_schema).invoke(
                messages,
                config={'tags': ['ollama', 'structured', self.model.name]},
                **kwargs)
        return response

    async def agenerate_structured_model(self,
                                         response_model: BaseModel,
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
            response_model.model_json_schema
        ).ainvoke(messages,
                  config={
                      'tags':
                      ['ollama', 'structured', self.model.name, 'async']
                  },
                  **kwargs)
        return response

    #retries if failed to parse the result as json
    def generate_json(self, prompt: dict[str, str], params=None, **kwargs):

        response_schemas = [
            ResponseSchema(name="title", description="The name of the item"),
            # ResponseSchema(name="description",
            #                description="A detailed description"),
            ResponseSchema(
                name="properties",
                description="List of properties as key-value pairs"),
            ResponseSchema(name="required",
                           description="List of required fields")
        ]

        # Create the output parser
        output_parser = StructuredOutputParser.from_response_schemas(
            response_schemas)

        chat_template = ChatPromptTemplate.from_messages([
            # ("system", prompt.get("system", "You are a helpful assistant.")),
            ("user", "{user_input}"),
        ])
        completion_chain = chat_template | self.model
        retry_parser = RetryOutputParser.from_llm(parser=output_parser,
                                                  llm=self.model,
                                                  max_retries=3)

        def parse_with_retry(x):
            return retry_parser.parse_with_prompt(
                prompt_value=x["prompt_value"],
                completion=x["completion"].content,
            )

        main_chain = RunnableParallel(
            completion=completion_chain,
            prompt_value=chat_template) | RunnableLambda(
                parse_with_retry) | RunnableLambda(kwargs.get("parser"))
        response = main_chain.invoke(
            {"user_input": prompt.get("user", "Answer my question")},
            config={'tags': ['ollama', 'json_mode', self.model.name]},
            **kwargs)

        import re

        def to_camel_case(s: str) -> str:
            # Remove non-alphanumeric characters and split words
            words = re.split(r'[^a-zA-Z0-9]', s)
            # Capitalize all words except the first one and join them
            return words[0].lower() + ''.join(
                word.capitalize() for word in words[1:]) if words else ''

        schema = {
            # "$schema": "http://json-schema.org/draft-07/schema#",
            "name": to_camel_case(response["title"]),
            # "type": "object",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": response["properties"],
                "required": response["required"]
            },
        }

        return json.dumps(schema)

    async def agenerate_json(self,
                             prompt: dict[str, str],
                             params=None,
                             **kwargs):
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
                completion=x["completion"].content,
            )

        main_chain = RunnableParallel(
            completion=completion_chain,
            prompt_value=chat_template) | RunnableLambda(
                parse_with_retry) | RunnableLambda(kwargs.get("parser"))
        response = await main_chain.ainvoke(
            {"user_input": prompt.get("user", "Answer my question")},
            config={'tags': ['ollama', 'json_mode', self.model.name]},
            **kwargs)
        return json.dumps(response)

    def generate_structured_schema(self,
                                   json_schema,
                                   prompt: dict[str, str] = None,
                                   method='json_mode',
                                   params=None,
                                   **kwargs):
        chat_template = ChatPromptTemplate.from_messages([
            ("system", prompt.get("system", "You are a helpful assistant.")),
            ("user", "{user_input}"),
        ])
        messages = chat_template.format_messages(
            user_input=prompt.get("user", "Answer my question"))
        response = self.model.with_structured_output(json_schema,
                                                     method=method).invoke(
                                                         messages, **kwargs)
        return response

    async def agenerate_structured_schema(self,
                                          json_schema,
                                          prompt: dict[str, str] = None,
                                          method='json_mode',
                                          params=None,
                                          **kwargs):
        chat_template = ChatPromptTemplate.from_messages([
            ("system", prompt.get("system", "You are a helpful assistant.")),
            ("user", "{user_input}"),
        ])
        messages = chat_template.format_messages(
            user_input=prompt.get("user", "Answer my question"))
        response = await self.model.with_structured_output(
            json_schema, method=method).ainvoke(messages, **kwargs)
        return response

    def generate_structured_model_from_image(self,
                                             image_bytes,
                                             response_model,
                                             prompt: dict[str, str] = None,
                                             params=None,
                                             **kwargs):
        image_base64 = base64.b64encode(image_bytes).decode()
        client = instructor.from_openai(OpenAI(base_url=self.config.base_url,
                                               api_key=self.config.api_key),
                                        mode=instructor.mode.Mode.JSON_SCHEMA)
        response = client.chat.completions.create(
            model=self.config.model,
            response_model=response_model,
            messages=[{
                "role":
                "system",
                "content":
                prompt.get("system", "You are a helpful assistant.")
            }, {
                "role":
                "user",
                "content": [{
                    "type": "text",
                    "text": prompt.get("user", "Tell me what you see")
                }, {
                    "type":
                    "image_url",
                    "image_url":
                    "data:image/jpeg;base64," + image_base64,
                }]
            }],
            **kwargs)
        return response

    async def agenerate_structured_model_from_image(self,
                                                    image_bytes,
                                                    response_model,
                                                    prompt: dict[str,
                                                                 str] = None,
                                                    params=None,
                                                    **kwargs):
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        client = instructor.apatch(AsyncOpenAI(base_url=self.config.base_url,
                                               api_key=self.config.api_key),
                                   mode=instructor.mode.Mode.JSON_SCHEMA)
        response = await client.chat.completions.create(
            model=self.config.model,
            response_model=response_model,
            messages=[{
                "role":
                "system",
                "content":
                prompt.get("system", "You are a helpful assistant.")
            }, {
                "role":
                "user",
                "content": [{
                    "type": "text",
                    "text": prompt.get("user", "Tell me what you see")
                }, {
                    "type":
                    "image_url",
                    "image_url":
                    "data:image/jpeg;base64," + image_base64,
                }]
            }],
            **kwargs)
        return response

    def generate_structured_schema_from_image(self,
                                              image_bytes,
                                              json_schema,
                                              prompt: dict[str, str] = None,
                                              method='json_schema',
                                              params=None,
                                              **kwargs):
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        chat_template = ChatPromptTemplate.from_messages([
            ("system", prompt.get("system", "Tell me what you see.")),
            MessagesPlaceholder("msg"),
        ])
        messages = chat_template.invoke({
            "msg": [
                HumanMessage(content=[{
                    "type": "text",
                    "text": "{user_input}"
                }, {
                    "type":
                    "image_url",
                    "image_url":
                    "data:image/jpeg;base64," + image_base64,
                }])
            ]
        }).to_messages()
        response = self.model.with_structured_output(json_schema,
                                                     method=method).invoke(
                                                         messages, **kwargs)
        return response

    async def agenerate_structured_schema_from_image(self,
                                                     image_bytes,
                                                     json_schema,
                                                     prompt: dict[str,
                                                                  str] = None,
                                                     method='json_schema',
                                                     params=None,
                                                     **kwargs):
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        chat_template = ChatPromptTemplate.from_messages([
            ("system", prompt.get("system", "Tell me what you see.")),
            MessagesPlaceholder("msg"),
        ])
        messages = chat_template.invoke({
            "msg": [
                HumanMessage(content=[{
                    "type": "text",
                    "text": "{user_input}"
                }, {
                    "type":
                    "image_url",
                    "image_url":
                    "data:image/jpeg;base64," + image_base64,
                }])
            ]
        }).to_messages()
        response = self.model.with_structured_output(json_schema,
                                                     method=method).ainvoke(
                                                         messages, **kwargs)
        return response
