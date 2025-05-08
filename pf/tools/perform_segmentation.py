from promptflow.core import tool

from functions.segment_document import segment


@tool
async def perform_segmentation(
    pages: list[str],
    json_schema: dict,
) -> list[dict]:
    segment_results = segment(json_schema, pages)
    return segment_results
