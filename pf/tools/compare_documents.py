from functions.comparison import compare_documents
from promptflow.core import tool
from promptflow.tracing import start_trace


@tool
async def compare_docs(comparison_key_sets: dict, ) -> dict:
    """
    Compare documents based on the provided key sets.
    """
    comparison_result = compare_documents(comparison_key_sets)
    return comparison_result
