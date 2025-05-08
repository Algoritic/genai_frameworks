from typing import List
from promptflow.core import tool

from functions.ocr import parallel_document_intelligence


@tool
async def perform_ocr(image_bytes: List[bytes]):
    meta = parallel_document_intelligence(image_bytes)
    return meta
