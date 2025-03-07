import asyncio
import base64
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
from typing import Any, Dict, Union, Optional
from core.logger import logger
import os
import tempfile
import pymupdf
import magic
import cv2
import numpy as np
from PIL import Image
import io


def get_file_mimetype(file: bytes) -> str | None:
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(file)
        temp_file.seek(0)
        logger.info("Guessing file type for %s" % temp_file.name)
        return magic.from_buffer(temp_file.read(), mime=True)


def batch_optimize_image(folder_path: str) -> str:  #return folder path
    #load all files under the folder
    files = os.listdir(folder_path)
    if len(files) == 0:
        raise ValueError("No files found in folder")
    for f in files:
        file_path = os.path.join(folder_path, f)
        #Image optimizations
        # optimizer = ImageOptimizer(profile_name="text_focused")
        # optimizer.load_image(file_path)
        # optimizer.optimize()
        # optimizer.save_final_result(file_path)

        image = cv2.imread(file_path)
        if (image is None):
            raise ValueError("Unable to open image")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced)
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 3.0)
        sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
        # _, binary = cv2.threshold(sharpened, 0, 255,
        #                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(file_path, sharpened)

        logger.info("Image optimized and saved to %s" % file_path)

        #image quality check
        # analyzer = ImageQualityAnalyzer("low_quality")
        # if analyzer.load_image(file_path):
        #     results = analyzer.analyze()
        #     # Print analysis results
        #     print("\n=== Image Quality Analysis Results ===")
        #     print(
        #         f"Overall Score: {results['overall_score']:.1f}/100 (Threshold: {results['overall_threshold']})"
        #     )
        #     print(f"OCR Ready: {'Yes' if results['passed'] else 'No'}")
        #     print("\nDetailed Metrics:")

        #     for name, metric in results["metrics"].items():
        #         status = "✓" if metric["passed"] else "✗"
        #         print(
        #             f"- {name.title()}: {metric['value']:.2f} {status} (Score: {metric['score']:.1f}/100)"
        #         )

        #     # Get preprocessing recommendations
        #     recommendations = analyzer.get_preprocessing_recommendations()
        #     print("\nRecommendations:")
        #     for key, recommendation in recommendations.items():
        #         print(f"- {key.title()}: {recommendation}")
        # else:
        #     print(f"Failed to load image: {file_path}")

    return folder_path


# def render_pdf_to_png(local_pdf_path: str,
#                       page_num: int,
#                       target_longest_image_dim: int = 2048) -> bytes:
#     longest_dim = max(get_pdf_media_box_width_height(local_pdf_path, page_num))

#     # Convert PDF page to PNG using pdftoppm
#     pdftoppm_result = subprocess.run(
#         [
#             "pdftoppm",
#             "-png",
#             "-f",
#             str(page_num),
#             "-l",
#             str(page_num),
#             "-r",
#             str(target_longest_image_dim * 72 /
#                 longest_dim),  # 72 pixels per point is the conversion factor
#             local_pdf_path,
#         ],
#         timeout=120,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#     )
#     assert pdftoppm_result.returncode == 0, pdftoppm_result.stderr
#     return pdftoppm_result.stdout


def compress_scanned_document(image_data: Union[str, bytes, np.ndarray],
                              output_path: Optional[str] = None,
                              dpi: int = 300,
                              quality: int = 85,
                              mode: str = "mixed",
                              max_size_mb: Optional[float] = None,
                              binary_threshold: int = 128,
                              target_format: str = "pdf") -> Dict[str, Any]:
    """
    Compress a scanned document image with optimizations specific to document content.

    Args:
        image_data: Path to image file, bytes of image, or numpy array
        output_path: Path to save compressed image (None returns bytes)
        dpi: Output DPI (dots per inch)
        quality: JPEG quality for continuous-tone regions (0-100)
        mode: Compression strategy: 'mixed' (text+image), 'text' (b&w), or 'image' (photos)
        max_size_mb: Maximum file size in MB (if set, will auto-adjust params to meet)
        binary_threshold: Threshold for binarization (0-255)
        target_format: Output format ('pdf', 'jpeg', 'png')

    Returns:
        Dict with compression results including compressed data and stats
    """
    # Load the image
    if isinstance(image_data, str):
        # It's a file path
        if not os.path.exists(image_data):
            raise FileNotFoundError(f"Image file not found: {image_data}")
        image = cv2.imread(image_data)
        if image is None:
            raise ValueError(f"Could not read image file: {image_data}")
        #delete the file
        # os.remove(image_data)
        original_size = os.path.getsize(image_data)
    elif isinstance(image_data, bytes):
        # It's bytes
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image bytes")
        original_size = len(image_data)
    elif isinstance(image_data, np.ndarray):
        # It's already a numpy array
        image = image_data.copy()
        # Estimate original size
        is_success, encoded_img = cv2.imencode('.png', image)
        original_size = len(encoded_img) if is_success else 0
    else:
        raise TypeError(
            "image_data must be a file path, bytes, or numpy array")

    # Convert to RGB (from BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    height, width = image.shape[:2]

    # Process based on mode
    if mode == "text" or mode == "mixed":
        # Preprocessing steps that work well for documents with text
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoisingColored(image_rgb, None, 10, 10, 7,
                                                   21)

        # 2. Convert to grayscale
        gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)

        # 3. Increase contrast using adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        if mode == "text":
            # 4. Binarization for pure text documents
            _, binary = cv2.threshold(enhanced, binary_threshold, 255,
                                      cv2.THRESH_BINARY)
            processed_image = binary
        else:  # mixed mode
            # Extract text regions using edge detection
            edges = cv2.Canny(enhanced, 100, 200)
            # Dilate to connect nearby edges (text regions)
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=2)

            # Create a mask for text regions
            text_mask = dilated_edges > 0

            # Apply binarization to text regions
            _, binary_text = cv2.threshold(enhanced, binary_threshold, 255,
                                           cv2.THRESH_BINARY)

            # Final image: binary text + enhanced grayscale for non-text
            processed_image = enhanced.copy()
            processed_image[text_mask] = binary_text[text_mask]
    else:  # image mode
        # For photograph-heavy documents, use standard image enhancement
        # 1. Denoise
        processed_image = cv2.fastNlMeansDenoisingColored(
            image_rgb, None, 10, 10, 7, 21)

    # Convert to PIL Image for saving with specific formats and DPI
    if len(processed_image.shape) == 2:  # Grayscale
        pil_image = Image.fromarray(processed_image)
    else:  # Color
        pil_image = Image.fromarray(processed_image)

    # Iterative compression to meet max_size constraint
    current_quality = quality
    compressed_data = None
    compressed_size = float('inf')

    # Try to compress to target size if max_size_mb is specified
    if max_size_mb is not None:
        target_size = int(max_size_mb * 1024 * 1024)  # Convert MB to bytes
        min_quality = 50 if mode != "text" else 75  # Lower bound on quality

        while current_quality >= min_quality and compressed_size > target_size:
            buf = io.BytesIO()

            if target_format.lower() == 'pdf':
                # For PDF, we need to save to a temporary file first
                with tempfile.NamedTemporaryFile(suffix='.pdf',
                                                 delete=False) as tmp:
                    pil_image.save(tmp.name,
                                   format='PDF',
                                   resolution=dpi,
                                   quality=current_quality)
                with open(tmp.name, 'rb') as f:
                    compressed_data = f.read()
                os.unlink(tmp.name)
            elif target_format.lower() == 'jpeg':
                pil_image.save(buf,
                               format='JPEG',
                               quality=current_quality,
                               dpi=(dpi, dpi),
                               optimize=True)
                compressed_data = buf.getvalue()
            elif target_format.lower() == 'png':
                # For PNG, we control compression level (0-9)
                compression_level = min(9, int(9 - (current_quality / 10)))
                pil_image.save(buf,
                               format='PNG',
                               dpi=(dpi, dpi),
                               compress_level=compression_level,
                               optimize=True)
                compressed_data = buf.getvalue()

            compressed_size = len(compressed_data)
            current_quality -= 5  # Reduce quality and try again

    # If we didn't need to iterate for size or failed to meet the target, compress once
    if compressed_data is None:
        buf = io.BytesIO()

        if target_format.lower() == 'pdf':
            # For PDF, we need to save to a temporary file first
            with tempfile.NamedTemporaryFile(suffix='.pdf',
                                             delete=False) as tmp:
                pil_image.save(tmp.name,
                               format='PDF',
                               resolution=dpi,
                               quality=quality)
            with open(tmp.name, 'rb') as f:
                compressed_data = f.read()
            os.unlink(tmp.name)
        elif target_format.lower() == 'jpeg':
            pil_image.save(buf,
                           format='JPEG',
                           quality=quality,
                           dpi=(dpi, dpi),
                           optimize=True)
            compressed_data = buf.getvalue()
        elif target_format.lower() == 'png':
            pil_image.save(buf, format='PNG', dpi=(dpi, dpi), optimize=True)
            compressed_data = buf.getvalue()
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

        compressed_size = len(compressed_data)

    # Save to file if output_path is provided
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(compressed_data)

    # Calculate compression ratio and percentage
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    compression_percentage = (
        1 -
        (compressed_size / original_size)) * 100 if original_size > 0 else 0

    return {
        "data": compressed_data,
        "compressed_size_bytes": compressed_size,
        "original_size_bytes": original_size,
        "compression_ratio": compression_ratio,
        "compression_percentage": compression_percentage,
        "width": width,
        "height": height,
        "quality": current_quality,
        "mode": mode,
        "format": target_format
    }


def batch_compress_documents(input_dir: str, output_dir: str,
                             **kwargs) -> Dict[str, Any]:
    """
    Batch compress all document images in a directory.

    Args:
        input_dir: Directory containing images to compress
        output_dir: Directory to save compressed images
        **kwargs: Additional arguments passed to compress_scanned_document

    Returns:
        Dict with summary statistics of the batch compression
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and any(
            f.lower().endswith(ext) for ext in image_extensions)
    ]

    if not image_files:
        return {"error": "No image files found in input directory"}

    results = []
    total_original_size = 0
    total_compressed_size = 0

    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)

        # Create output filename with the target format extension
        file_base = os.path.splitext(img_file)[0]
        target_format = kwargs.get('target_format', 'png')
        output_filename = f"{file_base}.{target_format.lower()}"
        output_path = os.path.join(output_dir, output_filename)

        try:
            result = compress_scanned_document(input_path, output_path,
                                               **kwargs)
            results.append({
                "filename":
                img_file,
                "output_filename":
                output_filename,
                "compression_ratio":
                result["compression_ratio"],
                "compression_percentage":
                result["compression_percentage"]
            })
            total_original_size += result["original_size_bytes"]
            total_compressed_size += result["compressed_size_bytes"]
        except Exception as e:
            results.append({"filename": img_file, "error": str(e)})

    # Calculate overall statistics
    overall_compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 0
    overall_compression_percentage = (
        1 - (total_compressed_size /
             total_original_size)) * 100 if total_original_size > 0 else 0

    return output_dir


def pdf_to_images(file: bytes,
                  file_ext: str,
                  first_page_only: bool = False) -> str:  #dir path
    file_type = get_file_mimetype(file)
    logger.info(file_type)
    output_folder = tempfile.mkdtemp()
    with tempfile.NamedTemporaryFile(
            dir=output_folder if file_ext != ".pdf" else None,
            suffix=file_ext,
            delete=False,
    ) as temp_file:
        temp_file.write(file)
        temp_file.seek(0)
        file_name = os.path.basename(temp_file.name)
        if file_type == "application/pdf":
            logger.info("Converting pdf to images")
            doc = pymupdf.open(temp_file.name)
            logger.info("total pages %s" % len(doc))
            for page in doc:
                logger.info("Saving %s" % output_folder)
                # png_bytes = render_pdf_to_png(temp_file.name, page.number)
                # #write to file
                # with open(
                #         f"{output_folder}/{file_name}-page-%i.png" %
                #         page.number, "wb") as f:
                #     f.write(png_bytes)
                pix = page.get_pixmap(dpi=400)
                pix.save(f"{output_folder}/{file_name}-page-%i.png" %
                         page.number)
                if first_page_only:
                    return output_folder
        else:
            logger.info("Image received")
            # temp_file.write(file)
            # temp_file.seek(0)
            return output_folder
    return output_folder
