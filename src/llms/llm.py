from abc import ABC, abstractmethod
from typing import Dict


class LLMBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def generate(self, prompt: str, params: Dict = None, **kwargs) -> str:
        pass
