from dataclasses import dataclass

from langchain.evaluation import load_evaluator

from promptflow.client import PFClient
from promptflow.tracing import trace

from llms.ollama_llm import OllamaLLM
from core.settings import app_settings


@dataclass
class Result:
    reasoning: str
    value: str
    score: float


class LangChainEvaluator:

    def __init__(self):
        self.llm = OllamaLLM(app_settings.ollama)
        # evaluate with langchain evaluator for conciseness
        self.evaluator = load_evaluator("criteria",
                                        llm=self.llm.model,
                                        criteria="conciseness")

    @trace
    def __call__(
        self,
        input: str,
        prediction: str,
    ) -> Result:
        """Evaluate with langchain evaluator."""

        eval_result = self.evaluator.evaluate_strings(prediction=prediction,
                                                      input=input)
        return Result(**eval_result)
