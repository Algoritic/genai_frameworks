from promptflow.core import tool
from processors.file_processor import batch_optimize_image


@tool
async def optimize_image(folder_path: str):
    folder_path = batch_optimize_image(folder_path)
    return folder_path
