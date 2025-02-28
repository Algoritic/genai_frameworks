import os
from promptflow.core import tool

from tools.ocr import advanced_pdf_ocr, simple_pdf_ocr, use_azure_document_intelligence, use_doctr, use_document_intelligence_llm_ocr, use_easy_ocr, use_vision_llm


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
            if ocr_strategy == "vision_llm":
                page_text += use_vision_llm(f.read())
            if ocr_strategy == "azure_document_intelligence":
                page_text += use_azure_document_intelligence(f.read())
            if ocr_strategy == "advanced":
                page_text += advanced_pdf_ocr(f.read())
            if ocr_strategy == "document_intelligence_llm":
                page_text += use_document_intelligence_llm_ocr(f.read())
    #clean up the folder
    # os.rmdir(folder_path)

    return page_text
