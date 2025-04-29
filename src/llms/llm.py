from abc import ABC, abstractmethod
from typing import Dict

from pydantic import BaseModel


class LLMBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def generate(self, prompt: str, params: Dict = None, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_structured_model(self,
                                  response_model: BaseModel,
                                  prompt: dict[str, str] = None,
                                  params=None,
                                  **kwargs) -> str:
        pass
