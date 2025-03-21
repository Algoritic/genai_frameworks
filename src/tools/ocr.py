import os
from pathlib import Path
import tempfile
import easyocr
import jinja2
import pymupdf
from llms.azure_llm import AzureLLM
from llms.mistral_llm import MistralLLM
from llms.ollama_llm import OllamaLLM
from llms.openai_llm import OAILLM
from processors.file_processor import get_file_mimetype
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from core.settings import app_settings
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature, DocumentContentFormat
from rapidocr import RapidOCR


def use_easy_ocr(bytes: bytes) -> str:
    reader = easyocr.Reader(['en'])
    result = reader.readtext(bytes, detail=0)
    return " ".join([r for r in result])


def use_doctr(path: str) -> str:
    predictor = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_images(path)
    result = predictor(doc)
    return result.render()


def use_local_vision_llm(bytes: bytes) -> str:
    llm = OllamaLLM(app_settings.ollama)
    result = llm.generate_from_image(image_bytes=bytes,
                                     prompt={
                                         "user":
                                         """
                You are a document extraction tool.
                Take a deep breath, look at this image and extract all the text content.
                You must:
                - Identify different sections or components.
                - Provide the output as plain text, maintaining the original layout and line breaks where appropriate.
                - Include all visible text from the image.
                - Read from left to right. Be concise.
                IMPORTANT: Do not include any introduction, explanation, or metadata. Only include the text content.
                """
                                     })
    return result


def use_vision_llm(bytes: bytes) -> str:
    llm = AzureLLM(app_settings.azure_openai)
    result = llm.generate_from_image(temperature=0,
                                     image_bytes=bytes,
                                     prompt={
                                         "user":
                                         """
                You are a document extraction tool.
                Take a deep breath, look at this image and extract all the text content.
                You must:
                - Identify different sections or components.
                - Provide the output as plain text, maintaining the original layout and line breaks where appropriate.
                - Include all visible text from the image.
                - Read from left to right. Be concise.
                IMPORTANT: Do not include any introduction, explanation, or metadata. Only include the text content.
                """
                                     },
                                     params={"detail": "high"})
    return result
    # result = llm.generate_structured_model_from_image(
    #     image_bytes=bytes,
    #     response_model=KeyValueList,
    #     prompt={
    #         "user":
    #         """You are a professional Document Extractor.
    #         You must
    #         - Identify different sections or components
    #         - Use appropriate keys for different text elements
    #         - Maintain the hierarchical structure of the content
    #         - Include all visible text from the image
    #         Extract every single key values from the following form. Read from left to right. Be concise."""
    #     },
    #     params={"details": "high"})

    # return result.model_dump_json()


def advanced_pdf_ocr(bytes: bytes, lang: list[str] = ['en']) -> str:
    os.environ[
        "TESSDATA_PREFIX"] = "/opt/homebrew/Cellar/tesseract/5.5.0/share/tessdata"
    page_text: str = ""  #serve as anchor text
    file_type = get_file_mimetype(bytes)
    if file_type is None:
        return "application/octet-stream"
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(bytes)
        temp_file.seek(0)
        #if file type if pdf, split into pages
        if file_type == "application/pdf":
            doc = pymupdf.open(temp_file.name)
            for page in doc:
                page_text += page.extract_text()
        if file_type == "image/jpeg" or file_type == "image/png":
            doc = pymupdf.open(temp_file.name)
            for page in doc:
                pix = page.get_pixmap()
                pdf = pix.pdfocr_tobytes()
                img_doc = pymupdf.open("pdf", pdf)
                pg = img_doc[0]
                page_text += pg.get_text()
        else:
            raise ValueError("Unsupported file type %s" % file_type)

    llm = AzureLLM(app_settings.azure_openai)
    root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent
    prompt_template_path = os.path.join(
        root_path,
        "prompts",
    )
    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(prompt_template_path))
    template = environment.get_template("doc_cleaning.jinja")
    prompt = template.render(base_text=page_text)
    result = llm.generate_from_image(
        temperature=0,
        image_bytes=bytes,
        # method="json_schema",
        # json_schema=openai_response_format_schema(),
        prompt={"user": prompt},
        params={"detail": "high"})

    return result


#file can be pdf or image
def simple_pdf_ocr(bytes: bytes, lang: list[str] = ['en']) -> str:
    os.environ[
        "TESSDATA_PREFIX"] = "/opt/homebrew/Cellar/tesseract/5.5.0/share/tessdata"
    page_text: str = ""
    file_type = get_file_mimetype(bytes)
    if file_type is None:
        return "application/octet-stream"
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(bytes)
        temp_file.seek(0)
        #if file type if pdf, split into pages
        if file_type == "application/pdf":
            doc = pymupdf.open(temp_file.name)
            for page in doc:
                page_text += page.extract_text()
        if file_type == "image/jpeg" or file_type == "image/png":
            doc = pymupdf.open(temp_file.name)
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                pdf = pix.pdfocr_tobytes()
                img_doc = pymupdf.open("pdf", pdf)
                pg = img_doc[0]
                page_text += pg.get_text()
        else:
            raise ValueError("Unsupported file type %s" % file_type)

    return page_text


def use_document_intelligence_llm_ocr(bytes: bytes) -> str:
    client = DocumentIntelligenceClient(
        endpoint=app_settings.azure_document_intelligence.endpoint,
        credential=AzureKeyCredential(
            app_settings.azure_document_intelligence.api_key))
    poller = client.begin_analyze_document(
        "prebuilt-layout",
        body=bytes,
        features=[
            # DocumentAnalysisFeature.KEY_VALUE_PAIRS,
            DocumentAnalysisFeature.OCR_HIGH_RESOLUTION
        ],
        output_content_format=DocumentContentFormat.TEXT,
    )
    result = poller.result()
    anchor_text = result.content
    llm = AzureLLM(app_settings.azure_openai)
    root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent
    prompt_template_path = os.path.join(
        root_path,
        "prompts",
    )
    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(prompt_template_path))
    template = environment.get_template("doc_cleaning.jinja")
    prompt = template.render(base_text=anchor_text)
    result = llm.generate_from_image(
        temperature=0.1,
        image_bytes=bytes,
        # method="json_schema",
        # json_schema=openai_response_format_schema(),
        prompt={"user": prompt},
        params={"detail": "high"})

    return result


def use_azure_document_intelligence(bytes: bytes) -> str:
    client = DocumentIntelligenceClient(
        endpoint=app_settings.azure_document_intelligence.endpoint,
        credential=AzureKeyCredential(
            app_settings.azure_document_intelligence.api_key))
    poller = client.begin_analyze_document(
        "prebuilt-layout",
        # "prebuilt-check.us",
        body=bytes,
        features=[
            # DocumentAnalysisFeature.KEY_VALUE_PAIRS,
            DocumentAnalysisFeature.OCR_HIGH_RESOLUTION
        ],
        output_content_format=DocumentContentFormat.TEXT,
    )
    result = poller.result()
    #calculate average, min, max confidence score
    average_confidence = 0
    min_confidence = 0
    max_confidence = 0
    word_count_list = [len(page.words) for page in result.pages]
    word_count = sum(word_count_list)
    confidences = [
        word.confidence for page in result.pages for word in page.words
    ]
    if len(confidences) > 0:
        average_confidence = round(sum(confidences) / len(confidences), 2)
        min_confidence = min(confidences)
        max_confidence = max(confidences)

    metrics = {
        "average_confidence": average_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
        "word_count": word_count
    }

    # combined_result = ""
    # print("----Key-value pairs found in document----")
    # if result.key_value_pairs:
    #     for kv_pair in result.key_value_pairs:
    #         if kv_pair.key:
    #             print(f"Key '{kv_pair.key.content}' found within ")
    #         if kv_pair.value:
    #             print(f"Value '{kv_pair.value.content}' found within ")
    return {
        "text": result.content,
        "metrics": metrics,
    }
    # for page in result.pages:
    #     for line in page.lines:
    #         combined_result += line.content
    # return combined_result


def use_mistral_ocr(bytes: bytes) -> str:
    llm = MistralLLM(config=app_settings.mistral)
    return llm.extract_pdf_text(bytes)


def use_openai(bytes: str) -> str:
    llm = OAILLM(app_settings.oai)
    result = llm.generate_from_image(image_bytes=bytes,
                                     prompt={
                                         "user":
                                         """
                You are a document extraction tool.
                Take a deep breath, look at this image and extract all the text content.
                You must:
                - Identify different sections or components.
                - Provide the output as plain text, maintaining the original layout and line breaks where appropriate.
                - Include all visible text from the image.
                - Read from left to right. Be concise.
                IMPORTANT: Do not include any introduction, explanation, or metadata. Only include the text content.
                """
                                     })
    return result


import numpy as np
from typing import List, Tuple, Union, Optional


def reorganize_ocr_output(
        dt_boxes: np.ndarray,
        txts: Optional[Union[List[str], Tuple[str]]] = None) -> str:
    """
    Reorganize OCR output to represent the actual document layout.

    Args:
        dt_boxes: numpy array of shape (n, 4, 2) containing bounding box coordinates
                 in the format [top-left, top-right, bottom-right, bottom-left]
        txts: list or tuple of strings corresponding to each bounding box

    Returns:
        A string representing the document with proper row-based layout
    """
    # Input validation
    if txts is None or len(dt_boxes) != len(txts):
        raise ValueError(
            "Number of bounding boxes and text strings must match")

    if len(dt_boxes) == 0:
        return ""

    # Calculate vertical range for each box
    box_info = []
    for i, box in enumerate(dt_boxes):
        # Extract y-coordinates
        y_coords = box[:, 1]

        # Find min and max y values
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)

        # Calculate center for sorting within rows
        center_y = (min_y + max_y) / 2

        # Calculate leftmost x-coordinate for sorting within a row
        x_coords = box[:, 0]
        min_x = np.min(x_coords)

        box_info.append({
            'id': i,
            'text': txts[i],
            'min_y': min_y,
            'max_y': max_y,
            'center_y': center_y,
            'min_x': min_x
        })

    # Group boxes into rows based on vertical overlap
    rows = []
    remaining_boxes = box_info.copy()

    while remaining_boxes:
        # Start a new row with the first box
        current_row = [remaining_boxes.pop(0)]
        current_min_y = current_row[0]['min_y']
        current_max_y = current_row[0]['max_y']

        # Calculate overlap threshold (adjustable parameter)
        threshold = 0.5 * (current_max_y - current_min_y)

        i = 0
        while i < len(remaining_boxes):
            box = remaining_boxes[i]

            # Check if this box overlaps with the current row
            overlap = min(current_max_y, box['max_y']) - max(
                current_min_y, box['min_y'])
            if overlap > threshold:
                # Add to current row
                current_row.append(box)

                # Update row's vertical range
                current_min_y = min(current_min_y, box['min_y'])
                current_max_y = max(current_max_y, box['max_y'])

                # Remove from remaining boxes
                remaining_boxes.pop(i)
            else:
                i += 1

        # Add the completed row
        rows.append(current_row)

    # Sort rows by center_y (top to bottom)
    rows.sort(key=lambda row: np.mean([box['center_y'] for box in row]))

    # Sort boxes within each row by min_x (left to right)
    for row in rows:
        row.sort(key=lambda box: box['min_x'])

    # Concatenate text with appropriate spacing
    result = []
    for row in rows:
        row_text = " ".join([box['text'] for box in row])
        result.append(row_text)

    # Join rows with newlines
    return "\n".join(result)


def use_rapid_ocr(path: str) -> str:
    engine = RapidOCR()
    result = engine(path)
    txts = result.txts
    result = reorganize_ocr_output(result.boxes, txts)
    return result
