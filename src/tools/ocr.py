import tempfile
import easyocr
import pymupdf
from llms.azure_llm import AzureLLM
from processors.file_processor import get_file_mimetype
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from core.settings import app_settings
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature, AnalyzeResult, DocumentContentFormat


def use_easy_ocr(bytes: bytes) -> str:
    reader = easyocr.Reader(['en'])
    result = reader.readtext(bytes, detail=0)
    return " ".join([r for r in result])


def use_doctr(path: str) -> str:
    predictor = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_images(path)
    result = predictor(doc)
    return result.render()


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


#file can be pdf or image
def simple_pdf_ocr(bytes: bytes, lang: list[str] = ['en']) -> str:
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


def use_azure_document_intelligence(bytes: bytes) -> str:
    client = DocumentIntelligenceClient(
        endpoint=app_settings.azure_document_intelligence.endpoint,
        credential=AzureKeyCredential(
            app_settings.azure_document_intelligence.api_key))
    poller = client.begin_analyze_document(
        "prebuilt-read",
        body=bytes,
        features=[
            # DocumentAnalysisFeature.KEY_VALUE_PAIRS,
            DocumentAnalysisFeature.OCR_HIGH_RESOLUTION
        ],
        output_content_format=DocumentContentFormat.TEXT,
    )
    result = poller.result()
    # combined_result = ""
    # print("----Key-value pairs found in document----")
    # if result.key_value_pairs:
    #     for kv_pair in result.key_value_pairs:
    #         if kv_pair.key:
    #             print(f"Key '{kv_pair.key.content}' found within ")
    #         if kv_pair.value:
    #             print(f"Value '{kv_pair.value.content}' found within ")
    return result.content
    # for page in result.pages:
    #     for line in page.lines:
    #         combined_result += line.content
    # return combined_result
