import subprocess
from typing import List
from core.logger import logger
import os
import tempfile
import pymupdf
import magic
import cv2

from tools.quality_validator import ImageQualityAnalyzer


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
        analyzer = ImageQualityAnalyzer("low_quality")
        if analyzer.load_image(file_path):
            results = analyzer.analyze()
            # Print analysis results
            print("\n=== Image Quality Analysis Results ===")
            print(
                f"Overall Score: {results['overall_score']:.1f}/100 (Threshold: {results['overall_threshold']})"
            )
            print(f"OCR Ready: {'Yes' if results['passed'] else 'No'}")
            print("\nDetailed Metrics:")

            for name, metric in results["metrics"].items():
                status = "✓" if metric["passed"] else "✗"
                print(
                    f"- {name.title()}: {metric['value']:.2f} {status} (Score: {metric['score']:.1f}/100)"
                )

            # Get preprocessing recommendations
            recommendations = analyzer.get_preprocessing_recommendations()
            print("\nRecommendations:")
            for key, recommendation in recommendations.items():
                print(f"- {key.title()}: {recommendation}")
        else:
            print(f"Failed to load image: {file_path}")

    return folder_path


def get_pdf_media_box_width_height(local_pdf_path: str,
                                   page_num: int) -> tuple[float, float]:
    """
    Get the MediaBox dimensions for a specific page in a PDF file using the pdfinfo command.

    :param pdf_file: Path to the PDF file
    :param page_num: The page number for which to extract MediaBox dimensions
    :return: A dictionary containing MediaBox dimensions or None if not found
    """
    # Construct the pdfinfo command to extract info for the specific page
    command = [
        "pdfinfo", "-f",
        str(page_num), "-l",
        str(page_num), "-box", "-enc", "UTF-8", local_pdf_path
    ]

    # Run the command using subprocess
    result = subprocess.run(command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)

    # Check if there is any error in executing the command
    if result.returncode != 0:
        raise ValueError(f"Error running pdfinfo: {result.stderr}")

    # Parse the output to find MediaBox
    output = result.stdout

    for line in output.splitlines():
        if "MediaBox" in line:
            media_box_str: List[str] = line.split(":")[1].strip().split()
            media_box: List[float] = [float(x) for x in media_box_str]
            return abs(media_box[0] - media_box[2]), abs(media_box[3] -
                                                         media_box[1])

    raise ValueError("MediaBox not found in the PDF info.")


def render_pdf_to_png(local_pdf_path: str,
                      page_num: int,
                      target_longest_image_dim: int = 2048) -> bytes:
    longest_dim = max(get_pdf_media_box_width_height(local_pdf_path, page_num))

    # Convert PDF page to PNG using pdftoppm
    pdftoppm_result = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            "-r",
            str(target_longest_image_dim * 72 /
                longest_dim),  # 72 pixels per point is the conversion factor
            local_pdf_path,
        ],
        timeout=120,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert pdftoppm_result.returncode == 0, pdftoppm_result.stderr
    return pdftoppm_result.stdout


def pdf_to_images(file: bytes, file_ext: str) -> str:  #dir path
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
        else:
            logger.info("Image received")
            # temp_file.write(file)
            # temp_file.seek(0)
            return output_folder
    return output_folder
