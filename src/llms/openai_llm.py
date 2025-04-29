import base64
from core.settings import OAISettings
from llms.llm import LLMBase
from langchain_openai import OpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage

from olmocr.pipeline import build_page_query


class OAILLM(LLMBase):

    def __init__(self, config: OAISettings, cache=None, model=None, **kwargs):
        super().__init__()

        self.config = config

        self.model = OpenAI(
            model=config.model,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
            # max_tokens=3000,
        )

    def generate(self, prompt, params=None, **kwargs):
        return super().generate(prompt, params, **kwargs)

    async def extract_pdf_text(self, file_path: str):
        query = await build_page_query(file_path,
                                       page=1,
                                       target_longest_image_dim=128,
                                       target_anchor_text_len=6000)
        query['model'] = self.config.model
        input = query["messages"]
        del query["messages"]
        response = self.model.invoke(input=input, **query)
        return response

    def generate_from_image(self,
                            image_bytes,
                            prompt: dict[str, str],
                            params=None,
                            **kwargs):
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        chat_template = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("msg"),
        ])
        messages = chat_template.invoke({
            "msg": [
                HumanMessage(
                    content=[{
                        "type": "text",
                        "text": prompt.get("user", "Answer my question")
                    }, {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + image_base64,
                        }
                    }])
            ]
        }).to_messages()
        response = self.model.invoke(messages, **kwargs)
        return response.content

    def generate_structured_model(self,
                                  response_model,
                                  prompt=None,
                                  params=None,
                                  **kwargs):
        return super().generate_structured_model(response_model, prompt,
                                                 params, **kwargs)
