from promptflow.core import tool
from processors.file_processor import batch_compress_documents


@tool
async def compress_image(folder_path: str):
    folder_path = batch_compress_documents(folder_path, folder_path)
    return folder_path
