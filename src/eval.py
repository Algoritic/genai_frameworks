from langchain.evaluation import load_evaluator
from langchain.evaluation import EvaluatorType
from core.settings import app_settings
from promptflow.client import PFClient
from llms.ollama_llm import OllamaLLM

# Get a pf client to manage runs
pf = PFClient()

# llm = OllamaLLM(app_settings.ollama)
# evaluator = load_evaluator(EvaluatorType.CRITERIA,
#                            criteria="conciseness",
#                            llm=llm.model)
# eval_result = evaluator.evaluate_strings(
#     prediction=
#     "What's 2+2? That's an elementary question. The answer you're looking for is that two and two is four.",
#     input="What's 2+2?",
# )
# print(eval_result)

data = "/Users/00153837yeohyihang/Desktop/Code/current-working/llm-framework/src/data.jsonl"  # path to the data file
# create run with the flow function and data
base_run = pf.run(
    flow=
    "/Users/00153837yeohyihang/Desktop/Code/current-working/llm-framework/src/flow.flex.yaml",
    # reference custom connection by name
    data=data,
    column_mapping={
        "prediction": "${data.prediction}",
        "input": "${data.input}",
    },
)
details = pf.get_details(base_run)
details.head(10)
pf.visualize([base_run])
