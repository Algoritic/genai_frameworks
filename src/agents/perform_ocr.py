import os
from promptflow.core import tool
import os
import asyncio
from typing import Dict, List, Any, Callable, Union
from tools.ocr import advanced_pdf_ocr, simple_pdf_ocr, use_azure_document_intelligence, use_doctr, use_document_intelligence_llm_ocr, use_easy_ocr, use_mistral_ocr, use_vision_llm


@tool
async def perform_ocr(folder_path: str, ocr_strategy: str) -> Dict[str, Any]:
    """
    Perform OCR on all files in a folder using the specified OCR strategy.

    Args:
        folder_path: Path to the folder containing files to process
        ocr_strategy: Name of the OCR strategy to use

    Returns:
        Dictionary containing extracted text, metrics, and page count
    """
    # Define a mapping of strategies to their corresponding functions
    strategy_functions = {
        "easy_ocr":
        lambda content, _: use_easy_ocr(content),
        "doctr":
        lambda _, file_path: use_doctr(file_path),
        "simple":
        lambda content, _: simple_pdf_ocr(content),
        "vision_llm":
        lambda content, _: use_vision_llm(content),
        "azure_document_intelligence":
        lambda content, _: use_azure_document_intelligence(content),
        "advanced":
        lambda content, _: advanced_pdf_ocr(content),
        "document_intelligence_llm":
        lambda content, _: use_document_intelligence_llm_ocr(content),
        "mistral":
        lambda content, _: use_mistral_ocr(content)
    }

    # Verify that the provided strategy is valid
    if ocr_strategy not in strategy_functions:
        raise ValueError(f"Unsupported OCR strategy: {ocr_strategy}")

    ocr_function = strategy_functions[ocr_strategy]

    # Gather all files in the directory
    file_paths = [
        f"{folder_path}/{f}" for f in os.listdir(folder_path)
        if os.path.isfile(f"{folder_path}/{f}")
    ]

    page_text_chunks = []
    metrics = []

    # Process files concurrently if possible
    if ocr_strategy in [
            "easy_ocr", "simple", "vision_llm", "azure_document_intelligence",
            "advanced", "document_intelligence_llm"
    ]:
        # These strategies can be processed concurrently
        async def process_file(file_path: str) -> Dict[str, Any]:
            with open(file_path, "rb") as f:
                content = f.read()
                file_name = f.name

                if ocr_strategy == "azure_document_intelligence":
                    result = ocr_function(content, file_name)
                    return {
                        "text": result['text'],
                        "metrics": result.get('metrics')
                    }
                else:
                    text = ocr_function(content, file_name)
                    return {"text": text}

        # Process files concurrently with semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(15)  # Limit to 5 concurrent operations

        async def bounded_process_file(file_path: str) -> Dict[str, Any]:
            async with semaphore:
                return await process_file(file_path)

        results = await asyncio.gather(
            *[bounded_process_file(file_path) for file_path in file_paths])

        for result in results:
            page_text_chunks.append(result["text"])
            if "metrics" in result and result["metrics"]:
                metrics.append(result["metrics"])
    else:
        # Sequential processing for strategies that can't be parallelized
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                content = f.read()
                text = ocr_function(content, f.name)
                page_text_chunks.append(text)

    # Join all text chunks efficiently
    page_text = "".join(page_text_chunks)

    # Clean up the folder if needed (commented out for safety)
    # Be cautious with this - only uncomment if you're sure
    # shutil.rmtree(folder_path)  # safer than os.rmdir which only works on empty dirs

    return {
        "text": page_text,
        "metrics": metrics,
        "page_count": len(file_paths)
    }
