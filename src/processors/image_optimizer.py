import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_image_for_ocr(
    image: np.ndarray,
    resize_width: Optional[int] = None,
    threshold_method: str = 'adaptive',
    denoise_strength: int = 10,
    border_size: int = 10,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Enhanced image preprocessing pipeline optimized for OCR.

    Args:
        image: Input image in BGR format
        resize_width: Target width to resize image (maintains aspect ratio)
        threshold_method: 'adaptive' or 'otsu'
        denoise_strength: Strength of denoising (higher = more aggressive)
        border_size: Size of border to add (helps with edge text)
        clahe_clip_limit: Contrast limit for CLAHE
        clahe_grid_size: Grid size for CLAHE

    Returns:
        Preprocessed image optimized for OCR
    """
    # Add border to help with edge text detection
    image = cv2.copyMakeBorder(image,
                               border_size,
                               border_size,
                               border_size,
                               border_size,
                               cv2.BORDER_CONSTANT,
                               value=[255, 255, 255])

    # Resize while maintaining aspect ratio
    if resize_width is not None:
        height, width = image.shape[:2]
        aspect_ratio = height / width
        new_height = int(resize_width * aspect_ratio)
        image = cv2.resize(image, (resize_width, new_height),
                           interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                            tileGridSize=clahe_grid_size)
    enhanced = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced,
                                        None,
                                        h=denoise_strength,
                                        searchWindowSize=21)

    # Sharpen using unsharp masking
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 3.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

    # Apply thresholding
    if threshold_method == 'adaptive':
        # Adaptive thresholding - good for varying lighting conditions
        binary = cv2.adaptiveThreshold(sharpened, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    else:  # 'otsu'
        # Otsu's thresholding - good for bimodal images
        _, binary = cv2.threshold(sharpened, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to remove noise and connect text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Deskew if needed
    angle = get_skew_angle(cleaned)
    if abs(angle) > 0.5:  # Only deskew if angle is significant
        cleaned = deskew_image(cleaned, angle)

    return cleaned


def get_skew_angle(image: np.ndarray) -> float:
    """
    Calculate skew angle of text in image.
    """
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = 90 + angle

    return angle


def deskew_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image to correct skew.
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image,
                             rotation_matrix, (width, height),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    return rotated
