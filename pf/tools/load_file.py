import re
from promptflow.core import tool
import os

from functions.pdf_parser import bytes_pdf_to_image_bytes, pdf_to_images
from logger import logger


@tool
async def load_file(
        file_path: str,
        use_in_memory: bool = False,
        first_page_only: bool = False):  #file path must be full path
    #only log the first 20 characters of the file path
    if len(file_path) > 20:
        logger.debug(f"File path: {file_path[:20]}...")
    else:
        logger.debug(f"File path: {file_path}")

    folder_path = None

    #check if the file path is remote or local file
    if file_path.startswith("http://") or file_path.startswith("https://"):
        # Handle remote file
        import requests
        response = requests.get(file_path)
        if response.status_code == 200:
            data = response.content
            folder_path = pdf_to_images(data, ".pdf", first_page_only)
        else:
            raise Exception(f"Failed to download file: {response.status_code}")
        return folder_path

    #check if the file path is a valid file path, using regex
    file_path_regex = r'^([a-zA-Z]:)?(\\|/)?(([\w\s().&\'-]+)(\\|/))*([\w\s().&\'-]+\.[\w]+)$'
    pattern = re.compile(file_path_regex)
    if pattern.fullmatch(file_path) is None:
        #fallback to treat file path as a base64 document payload
        #base64 to bytes
        import base64
        b64_str = file_path
        file_bytes = base64.b64decode(b64_str)
        # file_ext = ".pdf"
        folder_path = bytes_pdf_to_image_bytes(file_bytes)
        return folder_path

    else:
        if not os.path.isfile(file_path):
            raise Exception(f"File not found: {file_path}")
        # Handle local file
        with open(file_path, "rb") as f:
            file_ext = os.path.splitext(file_path)[1]
            data = f.read()
            folder_path = pdf_to_images(data, file_ext, first_page_only)
            return folder_path
