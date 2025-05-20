import concurrent
import time
from typing import List

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    DocumentAnalysisFeature,
    DocumentContentFormat,
)
from azure.core.credentials import AzureKeyCredential
from logger import logger
from promptflow.tracing import trace
from settings import app_settings


@trace
def use_azure_document_intelligence(
    bytes: bytes,
    model_name="prebuilt-layout",
    features=[DocumentAnalysisFeature.OCR_HIGH_RESOLUTION],
    api_version="2024-11-30",
) -> str:
    client = DocumentIntelligenceClient(
        api_version=api_version,
        endpoint=app_settings.azure_document_intelligence.endpoint,
        credential=AzureKeyCredential(app_settings.azure_document_intelligence.api_key),
    )
    poller = client.begin_analyze_document(
        model_id=model_name,
        body=bytes,
        features=features,
        output_content_format=DocumentContentFormat.TEXT,
    )
    result = poller.result()
    # calculate average, min, max confidence score
    average_confidence = 0
    min_confidence = 0
    max_confidence = 0
    word_count_list = [len(page.words) for page in result.pages]
    word_count = sum(word_count_list)
    confidences = [word.confidence for page in result.pages for word in page.words]
    if len(confidences) > 0:
        average_confidence = round(sum(confidences) / len(confidences), 2)
        min_confidence = min(confidences)
        max_confidence = max(confidences)

    metrics = {
        "average_confidence": average_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
        "word_count": word_count,
    }

    return {
        "text": result.content,
        "metrics": metrics,
    }


@trace
def parallel_document_intelligence(image_bytes: List[bytes]):
    batch_size = 15
    total_pages = len(image_bytes)
    results = [None] * total_pages
    metrics = [None] * total_pages

    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        batch_indices = list(range(batch_start, batch_end))
        batch_bytes = [image_bytes[i] for i in batch_indices]

        # Create a mapping function that tracks original indices
        def process_with_index(idx_path_tuple):
            idx, image_byte = idx_path_tuple
            return idx, use_azure_document_intelligence(image_byte)

        batch_start_time = time.time()
        # Process the current batch in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all jobs with their original indices
            future_to_idx = {
                executor.submit(process_with_index, (idx, image_byte)): idx
                for idx, image_byte in zip(batch_indices, batch_bytes)
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_idx):
                idx, ocr_result = future.result()
                results[idx] = ocr_result["text"]  # Store result at original index
                if "metrics" in ocr_result and ocr_result["metrics"]:
                    metrics[idx] = ocr_result["metrics"]

        # Calculate time to wait for rate limiting
        batch_processing_time = time.time() - batch_start_time
        if batch_processing_time < 1.0 and batch_end < total_pages:
            # Only wait if we processed faster than our rate limit and have more batches
            time.sleep(1.0 - batch_processing_time)

    page_text = "".join(results)

    logger.info("metrics: %s", metrics)

    return {
        "pages": results,
        "text": page_text,
        "metrics": metrics,
        "page_count": total_pages,
    }
