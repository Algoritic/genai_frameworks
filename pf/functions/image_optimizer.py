import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from logger import logger
from promptflow.tracing import trace


@trace
def optimize_image_fast(img_path: str) -> bool:
    """
    Speed-optimized single image processing function.

    Args:
        img_path: Path to image file

    Returns:
        bool: Success status
    """
    try:
        # Read image with optimized flag (IMREAD_REDUCED_GRAYSCALE_2 = read at 1/2 resolution)
        # This significantly reduces memory usage and processing time
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return False

        # Apply CLAHE for contrast enhancement (fastest method)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Use a faster bilateral filter instead of Non-Local Means
        # Bilateral is ~10x faster than NLMeans with similar quality
        denoised = cv2.bilateralFilter(enhanced, 5, 75, 75)

        # Skip Gaussian blur and directly apply unsharp mask
        # This combines two steps into one for speed
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Use optimized write with quality setting
        cv2.imwrite(img_path, sharpened, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return True

    except Exception:
        return False


@trace
def process_batch(file_batch: List[str]) -> Tuple[int, int]:
    """Process a batch of files, returns (success_count, total_count)"""
    success = 0
    for file_path in file_batch:
        if optimize_image_fast(file_path):
            success += 1
    return success, len(file_batch)


def batch_optimize_byte_images(files: List[bytes]) -> List[bytes]:
    # Calculate optimal batch size and worker count
    cpu_count = multiprocessing.cpu_count()
    batch_size = max(1, min(10, len(files) // cpu_count))
    worker_count = min(
        cpu_count * 2, len(files)
    )  # Use more threads than CPUs due to I/O operations
    # Split files into batches for better workload distribution
    batches = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]
    # Process using ThreadPoolExecutor (better for I/O bound tasks)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = executor.map(process_batch, batches)
    # Calculate success rate
    total_success = sum(r[0] for r in results)
    total_files = sum(r[1] for r in results)
    # Print stats only in non-production environments
    if os.environ.get("ENV") != "production":
        logger.info(
            f"Processed {total_success}/{total_files} images "
            f"({total_files / len(files):.1f} img/s)"
        )
    return files


@trace
def batch_optimize_image(folder_path: str) -> str:
    """
    Ultra-fast batch optimization of images in a folder.

    Args:
        folder_path: Path to folder with images

    Returns:
        str: Path to processed folder
    """
    # Convert to Path object
    path = Path(folder_path)

    # Fast file collection with list comprehension
    # Filter for common image extensions to avoid wasting time on non-images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [
        str(path / f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]

    if not files:
        raise ValueError("No image files found in folder")

    # Calculate optimal batch size and worker count
    cpu_count = multiprocessing.cpu_count()
    batch_size = max(1, min(10, len(files) // cpu_count))
    worker_count = min(
        cpu_count * 2, len(files)
    )  # Use more threads than CPUs due to I/O operations

    # Split files into batches for better workload distribution
    batches = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]

    # Process using ThreadPoolExecutor (better for I/O bound tasks)
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = executor.map(process_batch, batches)

    # Calculate success rate
    total_success = sum(r[0] for r in results)
    total_files = len(files)

    processing_time = time.time() - start_time
    # Print stats only in non-production environments
    if os.environ.get("ENV") != "production":
        logger.info(
            f"Processed {total_success}/{total_files} images in {processing_time:.2f}s "
            f"({total_files / processing_time:.1f} img/s)"
        )

    return folder_path
