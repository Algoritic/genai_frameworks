from core.settings import OAISettings
from llms.llm import LLMBase
from langchain_openai import OpenAI

from olmocr.pipeline import build_page_query


class OAILLM(LLMBase):

    def __init__(self, config: OAISettings, cache=None, model=None, **kwargs):
        super().__init__()

        self.config = config

        self.model = OpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=3000,
        )

    def generate(self, prompt, params=None, **kwargs):
        return super().generate(prompt, params, **kwargs)

    async def extract_pdf_text(self, file_path: str):
        query = await build_page_query(file_path,
                                       page=1,
                                       target_longest_image_dim=1024,
                                       target_anchor_text_len=6000)
        query['model'] = 'allenai_olmocr-7b-0225-preview'
        input = query["messages"]
        print(input)
        del query["messages"]
        response = self.model.invoke(input=input, **query)
        return response
