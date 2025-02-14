# flake8: noqa
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult
from deepeval.test_case import LLMTestCase


class DeepEvalCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs into deepeval.

    Args:
        implementation_name: name of the `implementation` in deepeval
        metrics: A list of metrics

    Raises:
        ImportError: if the `deepeval` package is not installed.

    Examples:
        >>> from langchain_community.llms import OpenAI
        >>> from langchain_community.callbacks import DeepEvalCallbackHandler
        >>> from deepeval.metrics import AnswerRelevancy
        >>> metric = AnswerRelevancy(minimum_score=0.3)
        >>> deepeval_callback = DeepEvalCallbackHandler(
        ...     implementation_name="exampleImplementation",
        ...     metrics=[metric],
        ... )
        >>> llm = OpenAI(
        ...     temperature=0,
        ...     callbacks=[deepeval_callback],
        ...     verbose=True,
        ...     openai_api_key="API_KEY_HERE",
        ... )
        >>> llm.generate([
        ...     "What is the best evaluation tool out there? (no bias at all)",
        ... ])
        "Deepeval, no doubt about it."
    """

    REPO_URL: str = "https://github.com/confident-ai/deepeval"
    ISSUES_URL: str = f"{REPO_URL}/issues"
    BLOG_URL: str = "https://docs.confident-ai.com"  # noqa: E501

    def __init__(
        self,
        metrics: List[Any],
        logger: Optional[logging.Logger] = None,
        implementation_name: Optional[str] = None,
    ) -> None:
        """Initializes the `deepevalCallbackHandler`.

        Args:
            implementation_name: Name of the implementation you want.
            metrics: What metrics do you want to track?

        Raises:
            ImportError: if the `deepeval` package is not installed.
            ConnectionError: if the connection to deepeval fails.
        """

        super().__init__()
        self.logger = logger

        # Import deepeval (not via `import_deepeval` to keep hints in IDEs)
        try:
            import deepeval  # ignore: F401,I001
        except ImportError:
            raise ImportError(
                """To use the deepeval callback manager you need to have the
                `deepeval` Python package installed. Please install it with
                `pip install deepeval`""")

        if os.path.exists(".deepeval"):
            warnings.warn(
                """You are currently not logging anything to the dashboard, we
                recommend using `deepeval login`.""")

        # Set the deepeval variables
        self.implementation_name = implementation_name
        self.metrics = metrics

        warnings.warn((
            "The `DeepEvalCallbackHandler` is currently in beta and is subject to"
            " change based on updates to `langchain`. Please report any issues to"
            f" {self.ISSUES_URL} as an `integration` issue."), )

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str],
                     **kwargs: Any) -> None:
        """Store the prompts"""
        self.prompts = prompts

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing when a new token is generated."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log records to deepeval when an LLM ends."""
        from deepeval.metrics import AnswerRelevancyMetric, BiasMetric, BaseMetric, ToxicityMetric, TaskCompletionMetric, PromptAlignmentMetric

        for metric in self.metrics:
            for i, generation in enumerate(response.generations):
                # Here, we only measure the first generation's output
                output = generation[0].text
                query = self.prompts[i]
                test_case = LLMTestCase(input=query, actual_output=output)
                if isinstance(metric, AnswerRelevancyMetric):
                    metric.measure(test_case)
                    self.logger.info(f"""
                                     Prompt: {query}\n
                                     Response: {output}\n
                                     Answer Relevancy: {metric.score}\n""")
                    print(f"Answer Relevancy: {metric.score}")  # noqa: T201
                elif isinstance(metric, TaskCompletionMetric):
                    metric.measure(test_case)
                    self.logger.info(f"""
                                     Prompt: {query}\n
                                     Response: {output}\n
                                     Task Completion: {metric.score}\n""")
                    print(f"Task Completion: {metric.score}")  # noqa: T201
                elif isinstance(metric, ToxicityMetric):
                    score = metric.measure(test_case)
                    self.logger.info(f"""
                                     Prompt: {query}\n
                                     Response: {output}\n
                                     Toxicity: {metric.score}\n""")
                    print(f"Toxic Score: {score}")  # noqa: T201
                elif isinstance(metric, BiasMetric):
                    metric.measure(test_case)
                    self.logger.info(f"""
                                     Prompt: {query}\n
                                     Response: {output}\n
                                     Bias: {metric.score}\n""")
                    print(f"Bias Score: {metric.score}")  # noqa: T201
                elif isinstance(metric, PromptAlignmentMetric):
                    metric.measure(test_case)
                    self.logger.info(f"""
                                     Prompt: {query}\n
                                     Response: {output}\n
                                     Prompt Alignment: {metric.score}\n""")
                    print(f"Prompt Alignment: {metric.score}")  # noqa: T201
                elif isinstance(metric, BaseMetric):
                    score = metric.measure(test_case)
                    print(f"Base Score: {score}")  # noqa: T201
                else:
                    raise ValueError(
                        f"""Metric {metric.__name__} is not supported by deepeval
                        callbacks.""")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Do nothing when LLM outputs an error."""
        pass

    def on_chain_start(self, serialized: Dict[str, Any],
                       inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Do nothing when chain starts"""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Do nothing when chain ends."""
        pass

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Do nothing when LLM chain outputs an error."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool starts."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Do nothing when agent takes a specific action."""
        pass

    def on_tool_end(
        self,
        output: Any,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool ends."""
        pass

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Do nothing when tool outputs an error."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Do nothing"""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Do nothing"""
        pass
