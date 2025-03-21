import os
import re
import shutil
import asyncio
from typing import Dict, List, Any, Callable, Union
from promptflow.core import tool
from tools.ocr import advanced_pdf_ocr, simple_pdf_ocr, use_azure_document_intelligence, use_doctr, use_document_intelligence_llm_ocr, use_easy_ocr, use_local_vision_llm, use_mistral_ocr, use_openai, use_rapid_ocr, use_vision_llm


@tool
async def perform_ocr(folder_path: str, ocr_strategy: str,
                      clean_up: bool) -> Dict[str, Any]:
    """
    Perform OCR on all files in a folder using the specified OCR strategy.
    Processes files in parallel while maintaining sequence order.

    Args:
        folder_path: Path to the folder containing files to process
        ocr_strategy: Name of the OCR strategy to use
        clean_up: Whether to delete the folder after processing

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
        "local_vision_llm":
        lambda content, _: use_local_vision_llm(content),
        "vision_llm":
        lambda content, _: use_vision_llm(content),
        "azure_document_intelligence":
        lambda content, _: use_azure_document_intelligence(content),
        "advanced":
        lambda content, _: advanced_pdf_ocr(content),
        "document_intelligence_llm":
        lambda content, _: use_document_intelligence_llm_ocr(content),
        "mistral":
        lambda content, _: use_mistral_ocr(content),
        "oai":
        lambda content, _: use_openai(content),
        "rapid_ocr":
        lambda content, _: use_rapid_ocr(content),
    }

    # Verify that the provided strategy is valid
    if ocr_strategy not in strategy_functions:
        raise ValueError(f"Unsupported OCR strategy: {ocr_strategy}")

    ocr_function = strategy_functions[ocr_strategy]

    # Gather all files in the directory and sort them to maintain sequence
    file_list = os.listdir(folder_path)
    # Sort files naturally (handles numeric sequences in filenames better)
    file_list.sort(
        key=lambda f:
        [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', f)])

    file_paths = [
        os.path.join(folder_path, f) for f in file_list
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    results = [None] * len(
        file_paths)  # Pre-allocate results list to maintain order
    metrics = []

    # All strategies can use parallel processing with proper ordering
    async def process_file(file_path: str, index: int) -> None:
        with open(file_path, "rb") as f:
            content = f.read()

            if ocr_strategy == "doctr":
                text = ocr_function(None, file_path)
            elif ocr_strategy == "azure_document_intelligence":
                result = ocr_function(content, f.name)
                text = result['text']
                if 'metrics' in result and result['metrics']:
                    metrics.append(result['metrics'])
            else:
                text = ocr_function(content, f.name)

            # Store result at correct position
            results[index] = text

    # Use semaphore to limit concurrent operations
    semaphore = asyncio.Semaphore(
        5
    )  # Limiting to 5 concurrent operations for better resource management

    async def bounded_process_file(file_path: str, index: int) -> None:
        async with semaphore:
            await process_file(file_path, index)

    # Create tasks for all files but maintain their indices
    tasks = [
        bounded_process_file(file_path, i)
        for i, file_path in enumerate(file_paths)
    ]

    # Run all tasks
    await asyncio.gather(*tasks)

    # Join all text chunks in correct order
    page_text = "".join(results)

    # Clean up the folder if needed
    if clean_up and os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    return {
        "text": page_text,
        "metrics": metrics,
        "page_count": len(file_paths)
    }
