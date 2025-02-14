from core.settings import EvalSettings
from deepeval.metrics import AnswerRelevancyMetric


class Evaluators:

    def __init__(self, settings: EvalSettings) -> None:
        self.settings = settings
        pass

    def eval_relevancy(self, prompt: str, response: str) -> float:
        metric = AnswerRelevancyMetric(
            threshold=self.settings.relevancy_threshold)
        test_case = ""
        pass
