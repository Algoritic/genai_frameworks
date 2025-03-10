import base64
import json
from mistralai import Mistral, ImageURLChunk
from core.settings import MistralSettings
from llms.llm import LLMBase


class MistralLLM(LLMBase):

    def __init__(self,
                 config: MistralSettings,
                 cache=None,
                 model=None,
                 **kwargs):
        super().__init__()

        self.config = config

        self.client = Mistral(api_key=config.api_key)

    def generate(self, prompt, params=None, **kwargs):
        return super().generate(prompt, params, **kwargs)

    def extract_pdf_text(self, image_bytes: bytes):
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        base64_data_url = f"data:image/jpeg;base64,{image_base64}"
        ocr_response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document=ImageURLChunk(image_url=base64_data_url),
            image_limit=1,
            retries=5,
            include_image_base64=True)
        response_dict = json.loads(ocr_response.model_dump_json())
        json_string = json.dumps(response_dict, indent=4)
        print("JJJJJ", json_string)
        return response_dict["pages"][0]["markdown"]
