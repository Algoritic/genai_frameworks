import os
from promptflow.core import tool

from processors.file_processor import pdf_to_images


@tool
async def load_file(file_path: str):  #file path must be full path
    with open(file_path, "rb") as f:
        file_ext = os.path.splitext(file_path)[1]
        data = f.read()
        folder_path = pdf_to_images(data, file_ext)
        return folder_path
