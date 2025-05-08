from typing import List
from promptflow.core import tool

from functions.image_optimizer import batch_optimize_byte_images


@tool
async def optimize_image(image_bytes: List[bytes]):
    output_image_bytes = batch_optimize_byte_images(image_bytes)
    return output_image_bytes
