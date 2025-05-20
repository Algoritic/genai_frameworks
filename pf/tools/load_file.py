import os
import re

from functions.pdf_parser import bytes_pdf_to_image_bytes, pdf_to_images
from logger import logger
from promptflow.core import tool


@tool
async def load_file(
    payload: str,
    use_in_memory: bool = False,
    extract_image: bool = True,
    first_page_only: bool = False,
):  # file path must be full path
    # only log the first 20 characters of the file path
    if len(payload) > 20:
        logger.debug(f"File path: {payload[:20]}...")
    else:
        logger.debug(f"File path: {payload}")

    folder_path = None

    # check if the file path is remote or local file
    if payload.startswith("http://") or payload.startswith("https://"):
        # Handle remote file
        import requests

        response = requests.get(payload)
        if response.status_code == 200:
            data = response.content
            folder_path = pdf_to_images(data, ".pdf", first_page_only, extract_image)
        else:
            raise Exception(f"Failed to download file: {response.status_code}")
        return folder_path

    # check if the file path is a valid file path, using regex
    file_path_regex = (
        r"^([a-zA-Z]:)?(\\|/)?(([\w\s().&\'-]+)(\\|/))*([\w\s().&\'-]+\.[\w]+)$"
    )
    pattern = re.compile(file_path_regex)
    if pattern.fullmatch(payload) is None:
        # fallback to treat file path as a base64 document payload
        # base64 to bytes
        import base64

        b64_str = payload

        # check if the string is base64 encoded
        if not re.match(r"^[A-Za-z0-9+/=]+$", b64_str):
            raise Exception(
                f"Invalid base64 string starting with: {b64_str[:10]}... Are you sending ocr text? Set skip_ocr to True."
            )

        file_bytes = base64.b64decode(b64_str)
        # file_ext = ".pdf"
        folder_path = bytes_pdf_to_image_bytes(file_bytes, extract_image)
        return folder_path

    else:
        if not os.path.isfile(payload):
            raise Exception(f"File not found: {payload}")
        # Handle local file
        with open(payload, "rb") as f:
            file_ext = os.path.splitext(payload)[1]
            data = f.read()
            folder_path = pdf_to_images(data, file_ext, first_page_only)
            return folder_path
