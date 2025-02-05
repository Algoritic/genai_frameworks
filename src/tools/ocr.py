import tempfile
import easyocr
import pymupdf
from data_models.generic import KeyValueList
from llms.azure_llm import AzureLLM
from processors.file_processor import get_file_mimetype
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from core.settings import app_settings


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
    llm = AzureLLM(app_settings.azure)
    result = llm.generate_from_image(
        image_bytes=bytes,
        prompt={
            "user":
            """Please look at this image and extract all the text content.
                Provide the output as plain text, maintaining the original layout and line breaks where appropriate.
                Include all visible text from the image."""
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
