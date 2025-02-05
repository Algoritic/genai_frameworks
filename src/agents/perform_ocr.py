import os
from promptflow.core import tool

from tools.ocr import simple_pdf_ocr, use_doctr, use_easy_ocr


@tool
async def perform_ocr(folder_path: str, ocr_strategy: str):
    page_text: str = ""
    for f in os.listdir(folder_path):
        with open(f"{folder_path}/{f}", "rb") as f:
            if ocr_strategy == "easy_ocr":
                page_text += use_easy_ocr(f.read())
            if ocr_strategy == "doctr":
                page_text += use_doctr(f.name)
            if ocr_strategy == "simple":
                page_text += simple_pdf_ocr(f.read())
    return page_text
