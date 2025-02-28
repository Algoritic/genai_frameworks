import cv2
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum
import time

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our image quality analyzer (assuming it's in the same directory)
try:
    from tools.quality_validator import ImageQualityAnalyzer, PresetManager
except ImportError:
    logger.error(
        "Could not import ImageQualityAnalyzer. Make sure ocr_image_quality.py is in the same directory."
    )

    # Define dummy classes for standalone use
    class ImageQualityAnalyzer:

        def __init__(self, preset_name="standard"):
            self.preset_name = preset_name

        def load_image(self, image_path):
            return True

        def analyze(self):
            return {"passed": False, "overall_score": 50}

        def get_preprocessing_recommendations(self):
            return {}

    class PresetManager:

        def list_presets(self):
            return ["standard"]


class OptimizationLevel(Enum):
    """Enum for optimization strength levels"""
    MILD = 1  # Conservative changes
    MODERATE = 2  # Balanced approach
    AGGRESSIVE = 3  # Strong corrections


class OptimizationProfile:
    """Class to define optimization profiles and parameters"""

    def __init__(self,
                 name: str,
                 description: str,
                 level: OptimizationLevel = OptimizationLevel.MODERATE):
        self.name = name
        self.description = description
        self.level = level
        self.params = {}

    def set_param(self, technique: str, param_name: str, value):
        """Set a parameter for a specific optimization technique"""
        if technique not in self.params:
            self.params[technique] = {}
        self.params[technique][param_name] = value

    def get_param(self, technique: str, param_name: str, default=None):
        """Get a parameter value with default fallback"""
        return self.params.get(technique, {}).get(param_name, default)


class ProfileManager:
    """Class to manage different optimization profiles"""

    def __init__(self):
        self.profiles = {}
        self._initialize_default_profiles()

    def _initialize_default_profiles(self):
        """Initialize default optimization profiles"""
        # Standard profile (moderate changes)
        standard = OptimizationProfile(
            "standard", "Balanced optimization for typical documents",
            OptimizationLevel.MODERATE)
        standard.set_param("contrast", "clip_limit", 2.0)
        standard.set_param("contrast", "tile_grid_size", (8, 8))
        standard.set_param("brightness", "gamma", 1.0)
        standard.set_param("brightness", "target_mean", 0.5)
        standard.set_param("sharpening", "amount", 1.5)
        standard.set_param("sharpening", "radius", 1.0)
        standard.set_param("sharpening", "threshold", 0)
        standard.set_param("noise", "h", 10)  # Non-local means h parameter
        standard.set_param("noise", "template_size", 7)
        standard.set_param("noise", "search_size", 21)
        standard.set_param("deskew", "max_angle", 45)
        standard.set_param("deskew", "angle_step", 0.5)
        standard.set_param("binarization", "block_size", 11)
        standard.set_param("binarization", "c", 2)
        self.profiles["standard"] = standard

        # Mild profile (conservative changes)
        mild = OptimizationProfile(
            "mild", "Conservative optimization to maintain authenticity",
            OptimizationLevel.MILD)
        mild.set_param("contrast", "clip_limit", 1.5)
        mild.set_param("contrast", "tile_grid_size", (8, 8))
        mild.set_param("brightness", "gamma", 1.0)
        mild.set_param("brightness", "target_mean", 0.5)
        mild.set_param("sharpening", "amount", 1.2)
        mild.set_param("sharpening", "radius", 0.8)
        mild.set_param("sharpening", "threshold", 3)
        mild.set_param("noise", "h", 15)  # Higher h = more smoothing
        mild.set_param("noise", "template_size", 7)
        mild.set_param("noise", "search_size", 21)
        mild.set_param("deskew", "max_angle", 30)
        mild.set_param("deskew", "angle_step", 1.0)
        mild.set_param("binarization", "block_size", 15)
        mild.set_param("binarization", "c", 5)
        self.profiles["mild"] = mild

        # Aggressive profile (strong corrections)
        aggressive = OptimizationProfile(
            "aggressive", "Strong optimization for poor quality images",
            OptimizationLevel.AGGRESSIVE)
        aggressive.set_param("contrast", "clip_limit", 3.0)
        aggressive.set_param("contrast", "tile_grid_size", (8, 8))
        aggressive.set_param("brightness", "gamma", 1.0)
        aggressive.set_param("brightness", "target_mean", 0.55)
        aggressive.set_param("sharpening", "amount", 2.0)
        aggressive.set_param("sharpening", "radius", 1.5)
        aggressive.set_param("sharpening", "threshold", 0)
        aggressive.set_param("noise", "h", 5)  # Lower h = less smoothing
        aggressive.set_param("noise", "template_size", 7)
        aggressive.set_param("noise", "search_size", 21)
        aggressive.set_param("deskew", "max_angle", 45)
        aggressive.set_param("deskew", "angle_step", 0.2)
        aggressive.set_param("binarization", "block_size", 9)
        aggressive.set_param("binarization", "c", 1)
        self.profiles["aggressive"] = aggressive

        # Text-focused profile (optimized for text extraction)
        text_focused = OptimizationProfile(
            "text_focused", "Optimized specifically for text extraction",
            OptimizationLevel.MODERATE)
        text_focused.set_param("contrast", "clip_limit", 2.5)
        text_focused.set_param("contrast", "tile_grid_size", (8, 8))
        text_focused.set_param("brightness", "gamma", 1.1)
        text_focused.set_param("brightness", "target_mean",
                               0.6)  # Slightly brighter for text
        text_focused.set_param("sharpening", "amount", 1.8)
        text_focused.set_param("sharpening", "radius", 1.0)
        text_focused.set_param("sharpening", "threshold", 0)
        text_focused.set_param("noise", "h", 10)
        text_focused.set_param("noise", "template_size", 7)
        text_focused.set_param("noise", "search_size", 21)
        text_focused.set_param("deskew", "max_angle", 45)
        text_focused.set_param("deskew", "angle_step", 0.5)
        text_focused.set_param("binarization", "block_size", 11)
        text_focused.set_param("binarization", "c", 2)
        text_focused.set_param("binarization", "use_sauvola",
                               True)  # Use Sauvola for text
        text_focused.set_param("binarization", "sauvola_k", 0.1)
        text_focused.set_param("binarization", "sauvola_window", 15)
        self.profiles["text_focused"] = text_focused

    def get_profile(self, name: str = "standard") -> OptimizationProfile:
        """Get a profile by name, defaults to standard if not found"""
        return self.profiles.get(name, self.profiles["standard"])

    def list_profiles(self) -> List[str]:
        """List all available profiles"""
        return list(self.profiles.keys())

    def add_profile(self, profile: OptimizationProfile):
        """Add a new profile"""
        self.profiles[profile.name] = profile


class OptimizationStep:
    """Class to track individual optimization steps and their results"""

    def __init__(self, technique: str, description: str,
                 input_image: np.ndarray):
        self.technique = technique
        self.description = description
        self.input_image = input_image.copy()
        self.output_image = None
        self.start_time = time.time()
        self.duration = 0
        self.params = {}
        self.success = False
        self.improvement_score = 0.0  # Quality improvement score

    def complete(self,
                 output_image: np.ndarray,
                 success: bool = True,
                 params: Dict = None):
        """Mark step as complete with results"""
        self.output_image = output_image
        self.duration = time.time() - self.start_time
        self.success = success
        if params:
            self.params = params

    def get_before_after(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get before and after images for comparison"""
        return self.input_image, self.output_image


class ImageOptimizer:
    """Main class for image optimization pipeline"""

    def __init__(self,
                 analyzer: Optional[ImageQualityAnalyzer] = None,
                 profile_name: str = "standard"):
        # Set up managers
        self.profile_manager = ProfileManager()
        self.profile = self.profile_manager.get_profile(profile_name)

        # Set analyzer or create a new one
        if analyzer:
            self.analyzer = analyzer
        else:
            preset_manager = PresetManager()
            self.analyzer = ImageQualityAnalyzer("standard")

        # Image states
        self.original_image = None
        self.current_image = None
        self.optimized_image = None
        self.gray_image = None

        # Results tracking
        self.steps = []
        self.initial_analysis = None
        self.final_analysis = None

        # Processing flags
        self.interactive = False
        self.save_intermediate = False
        self.intermediate_dir = "intermediate"

    def set_profile(self, profile_name: str):
        """Change the current optimization profile"""
        self.profile = self.profile_manager.get_profile(profile_name)
        logger.info(f"Changed optimization profile to: {profile_name}")

    def load_image(self, image_path: str) -> bool:
        """Load an image from file"""
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return False

        try:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                logger.error(f"Failed to read image: {image_path}")
                return False

            self.current_image = self.original_image.copy()
            self.gray_image = cv2.cvtColor(self.original_image,
                                           cv2.COLOR_BGR2GRAY)

            # Load image in analyzer too
            self.analyzer.load_image(image_path)

            logger.info(f"Successfully loaded image: {image_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return False

    def load_image_from_array(self, image: np.ndarray) -> bool:
        """Load image from numpy array"""
        try:
            self.original_image = image.copy()
            self.current_image = image.copy()

            if len(image.shape) == 3:
                self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                self.gray_image = image.copy()
                # Convert to 3 channel for consistency
                self.original_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                self.current_image = self.original_image.copy()

            logger.info("Successfully loaded image from array")
            return True
        except Exception as e:
            logger.error(f"Error loading image from array: {str(e)}")
            return False

    def _save_intermediate(self, image: np.ndarray, step_name: str):
        """Save intermediate results if enabled"""
        if self.save_intermediate:
            if not os.path.exists(self.intermediate_dir):
                os.makedirs(self.intermediate_dir)

            filename = os.path.join(self.intermediate_dir,
                                    f"{len(self.steps):02d}_{step_name}.png")
            cv2.imwrite(filename, image)
            logger.debug(f"Saved intermediate result: {filename}")

    def _interactive_approval(self, step: OptimizationStep) -> bool:
        """Get user approval for a step in interactive mode"""
        if not self.interactive:
            return True

        # Show before/after comparison
        before, after = step.get_before_after()

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        if len(before.shape) == 3:
            plt.imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(before, cmap='gray')
        plt.title("Before")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        if len(after.shape) == 3:
            plt.imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(after, cmap='gray')
        plt.title("After")
        plt.axis('off')

        plt.suptitle(f"{step.technique}: {step.description}")
        plt.tight_layout()
        plt.show()

        # Get user input
        while True:
            response = input("\nApply this change? (y/n): ").lower().strip()
            if response in ('y', 'yes'):
                return True
            elif response in ('n', 'no'):
                return False
            else:
                print("Please enter 'y' or 'n'")

    # ------ Optimization Techniques ------ #

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        step = OptimizationStep("contrast", "Contrast Enhancement", image)

        try:
            # Get parameters from profile
            clip_limit = self.profile.get_param("contrast", "clip_limit", 2.0)
            tile_grid_size = self.profile.get_param("contrast",
                                                    "tile_grid_size", (8, 8))

            # Work with grayscale for contrast enhancement
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                        tileGridSize=tile_grid_size)
                cl = clahe.apply(l)

                # Merge channels
                enhanced_lab = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                        tileGridSize=tile_grid_size)
                enhanced = clahe.apply(image)

            step.complete(enhanced, True, {
                "clip_limit": clip_limit,
                "tile_grid_size": tile_grid_size
            })
            self._save_intermediate(enhanced, "contrast")

            # Get approval in interactive mode
            approved = self._interactive_approval(step)
            if not approved:
                enhanced = image.copy()
                step.success = False

            self.steps.append(step)
            return enhanced
        except Exception as e:
            logger.error(f"Error in contrast enhancement: {str(e)}")
            step.complete(image.copy(), False)
            self.steps.append(step)
            return image.copy()

    def _correct_brightness(self, image: np.ndarray) -> np.ndarray:
        """Correct image brightness using gamma correction or normalization"""
        step = OptimizationStep("brightness", "Brightness Correction", image)

        try:
            # Get parameters from profile
            gamma = self.profile.get_param("brightness", "gamma", 1.0)
            target_mean = self.profile.get_param("brightness", "target_mean",
                                                 0.5)

            # Convert to grayscale if color
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Calculate current brightness
            current_mean = np.mean(gray) / 255.0

            # Determine if we need gamma correction based on current brightness
            if abs(current_mean - target_mean) > 0.1:
                # Calculate adjusted gamma to reach target brightness
                if current_mean > 0:
                    adjusted_gamma = np.log(target_mean) / np.log(current_mean)
                else:
                    adjusted_gamma = gamma

                # Apply gamma correction
                if len(image.shape) == 3:
                    # Process each channel with gamma correction
                    corrected = np.zeros_like(image, dtype=np.float32)
                    for i in range(3):
                        corrected[:, :, i] = (
                            (image[:, :, i] / 255.0)**adjusted_gamma) * 255.0

                    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
                else:
                    # Process grayscale
                    corrected = np.power(gray / 255.0, adjusted_gamma) * 255.0
                    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            else:
                corrected = image.copy()

            step.complete(
                corrected, True, {
                    "gamma":
                    gamma,
                    "target_mean":
                    target_mean,
                    "applied_gamma":
                    adjusted_gamma if abs(current_mean -
                                          target_mean) > 0.1 else 1.0,
                    "initial_brightness":
                    current_mean,
                    "final_brightness":
                    np.mean(corrected if len(corrected.shape) < 3 else cv2.
                            cvtColor(corrected, cv2.COLOR_BGR2GRAY)) / 255.0
                })
            self._save_intermediate(corrected, "brightness")

            # Get approval in interactive mode
            approved = self._interactive_approval(step)
            if not approved:
                corrected = image.copy()
                step.success = False

            self.steps.append(step)
            return corrected
        except Exception as e:
            logger.error(f"Error in brightness correction: {str(e)}")
            step.complete(image.copy(), False)
            self.steps.append(step)
            return image.copy()

    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image using unsharp masking"""
        step = OptimizationStep("sharpening", "Image Sharpening", image)

        try:
            # Get parameters from profile
            amount = self.profile.get_param("sharpening", "amount", 1.5)
            radius = self.profile.get_param("sharpening", "radius", 1.0)
            threshold = self.profile.get_param("sharpening", "threshold", 0)

            # Convert kernel radius to size (must be odd)
            kernel_size = int(2 * radius + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Create a blurred version for unsharp masking
            if len(image.shape) == 3:
                blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size),
                                           0)
                sharpened = cv2.addWeighted(image, 1.0 + amount, blurred,
                                            -amount, 0)
            else:
                blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size),
                                           0)
                sharpened = cv2.addWeighted(image, 1.0 + amount, blurred,
                                            -amount, 0)

            # Apply threshold if specified (only sharpen edges above threshold)
            if threshold > 0:
                # Calculate absolute difference for thresholding
                diff = cv2.absdiff(image, blurred)
                mask = diff > threshold

                # Apply sharpening only where threshold is exceeded
                sharpened = np.where(mask, sharpened, image)

            step.complete(
                sharpened, True, {
                    "amount": amount,
                    "radius": radius,
                    "kernel_size": kernel_size,
                    "threshold": threshold
                })
            self._save_intermediate(sharpened, "sharpening")

            # Get approval in interactive mode
            approved = self._interactive_approval(step)
            if not approved:
                sharpened = image.copy()
                step.success = False

            self.steps.append(step)
            return sharpened
        except Exception as e:
            logger.error(f"Error in image sharpening: {str(e)}")
            step.complete(image.copy(), False)
            self.steps.append(step)
            return image.copy()

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Reduce image noise using appropriate filtering"""
        step = OptimizationStep("noise", "Noise Reduction", image)

        try:
            # Get parameters from profile
            h = self.profile.get_param("noise", "h", 10)  # Filter strength
            template_size = self.profile.get_param("noise", "template_size", 7)
            search_size = self.profile.get_param("noise", "search_size", 21)

            # Use different strategies based on optimization level
            level = self.profile.level

            if len(image.shape) == 3:
                if level == OptimizationLevel.MILD:
                    # Bilateral filter (preserves edges better)
                    denoised = cv2.bilateralFilter(image, 9, 75, 75)
                elif level == OptimizationLevel.AGGRESSIVE:
                    # Non-local means denoising (stronger but slower)
                    denoised = cv2.fastNlMeansDenoisingColored(
                        image, None, h, h, template_size, search_size)
                else:
                    # Moderate level - median blur good for salt and pepper noise
                    denoised = cv2.medianBlur(image, 3)
            else:
                if level == OptimizationLevel.MILD:
                    # Bilateral filter (preserves edges better)
                    denoised = cv2.bilateralFilter(image, 9, 75, 75)
                elif level == OptimizationLevel.AGGRESSIVE:
                    # Non-local means denoising (stronger but slower)
                    denoised = cv2.fastNlMeansDenoising(
                        image, None, h, template_size, search_size)
                else:
                    # Moderate level - median blur
                    denoised = cv2.medianBlur(image, 3)

            step.complete(
                denoised, True, {
                    "method": level.name,
                    "h": h,
                    "template_size": template_size,
                    "search_size": search_size
                })
            self._save_intermediate(denoised, "denoising")

            # Get approval in interactive mode
            approved = self._interactive_approval(step)
            if not approved:
                denoised = image.copy()
                step.success = False

            self.steps.append(step)
            return denoised
        except Exception as e:
            logger.error(f"Error in noise reduction: {str(e)}")
            step.complete(image.copy(), False)
            self.steps.append(step)
            return image.copy()

    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew"""
        step = OptimizationStep("deskew", "Skew Correction", image)

        try:
            # Get parameters from profile
            max_angle = self.profile.get_param("deskew", "max_angle", 45)
            angle_step = self.profile.get_param("deskew", "angle_step", 0.5)

            # Create grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Threshold the image
            _, thresh = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find all contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # Find text lines using contours
            angles = []
            for c in contours:
                # Skip small contours
                if cv2.contourArea(c) < 100:
                    continue

                # Calculate minimum area rectangle
                rect = cv2.minAreaRect(c)
                # Get angle
                angle = rect[2]

                # Adjust angle range
                if angle < -45:
                    angle += 90

                # Keep angles in reasonable range
                if abs(angle) <= max_angle:
                    angles.append(angle)

            # No valid angles found
            if not angles:
                logger.info("No significant skew detected")
                step.complete(image.copy(), False, {"skew_angle": 0})
                self.steps.append(step)
                return image.copy()

            # Use median of angles to reduce outlier influence
            skew_angle = np.median(angles)

            # Only correct if skew angle is significant
            if abs(skew_angle) < 0.5:
                logger.info(
                    f"Skew angle too small to correct: {skew_angle:.2f}°")
                step.complete(image.copy(), False, {"skew_angle": skew_angle})
                self.steps.append(step)
                return image.copy()

            # Get image dimensions
            h, w = image.shape[:2]
            center = (w // 2, h // 2)

            # Calculate rotation matrix
            M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)

            # Calculate new image dimensions to avoid cropping
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # Adjust matrix for new dimensions
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            # Apply rotation
            deskewed = cv2.warpAffine(
                image,
                M, (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255) if len(image.shape) == 3 else 255)

            step.complete(deskewed, True, {"skew_angle": skew_angle})
            self._save_intermediate(deskewed, "deskew")

            # Get approval in interactive mode
            approved = self._interactive_approval(step)
            if not approved:
                deskewed = image.copy()
                step.success = False

            self.steps.append(step)
            return deskewed
        except Exception as e:
            logger.error(f"Error in image deskewing: {str(e)}")
            step.complete(image.copy(), False)
            self.steps.append(step)
            return image.copy()

    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Binarize image for text extraction"""
        step = OptimizationStep("binarization", "Image Binarization", image)

        try:
            # Get parameters from profile
            block_size = self.profile.get_param("binarization", "block_size",
                                                11)
            c = self.profile.get_param("binarization", "c", 2)
            use_sauvola = self.profile.get_param("binarization", "use_sauvola",
                                                 False)
            sauvola_k = self.profile.get_param("binarization", "sauvola_k",
                                               0.1)
            sauvola_window = self.profile.get_param("binarization",
                                                    "sauvola_window", 15)

            # Ensure block_size is odd
            if block_size % 2 == 0:
                block_size += 1

            # Create grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            if use_sauvola:
                # Implementation of Sauvola thresholding
                # Calculate mean and standard deviation in local windows
                mean = cv2.blur(gray, (sauvola_window, sauvola_window))
                mean_sq = cv2.blur(np.square(gray.astype(np.float32)),
                                   (sauvola_window, sauvola_window))
                std = np.sqrt(mean_sq - np.square(mean))

                # Calculate threshold using Sauvola formula
                threshold = mean * (1 + sauvola_k * ((std / 128) - 1))
                binary = np.zeros_like(gray)
                binary[gray > threshold] = 255
            else:
                # Standard adaptive thresholding
                binary = cv2.adaptiveThreshold(gray, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, block_size,
                                               c)

            step.complete(
                binary, True, {
                    "method":
                    "sauvola" if use_sauvola else "adaptive_gaussian",
                    "block_size": block_size,
                    "c": c,
                    "sauvola_k": sauvola_k if use_sauvola else None,
                    "sauvola_window": sauvola_window if use_sauvola else None
                })
            self._save_intermediate(binary, "binarization")

            # Get approval in interactive mode
            approved = self._interactive_approval(step)
            if not approved:
                binary = gray.copy()
                step.success = False

            self.steps.append(step)
            return binary
        except Exception as e:
            logger.error(f"Error in image binarization: {str(e)}")
            step.complete(image.copy(), False)
            self.steps.append(step)
            return image.copy()

    # ------ Image Optimization Pipeline ------ #

    def optimize(self) -> np.ndarray:
        """Run the full image optimization pipeline"""
        if self.current_image is None:
            logger.error("No image loaded for optimization")
            return None

        # Reset steps and results
        self.steps = []
        self.initial_analysis = None
        self.final_analysis = None

        # Start with original image
        self.optimized_image = self.current_image.copy()

        # Run each optimization step
        self.optimized_image = self._enhance_contrast(self.optimized_image)
        self.optimized_image = self._correct_brightness(self.optimized_image)
        self.optimized_image = self._sharpen_image(self.optimized_image)
        self.optimized_image = self._reduce_noise(self.optimized_image)
        self.optimized_image = self._deskew_image(self.optimized_image)
        self.optimized_image = self._binarize_image(self.optimized_image)

        # # Final analysis after optimization
        # self.final_analysis = self.analyzer.analyze_image(self.optimized_image)

        return self.optimized_image

    def get_step_results(self) -> List[Dict]:
        """Get results for each optimization step"""
        results = []
        for step in self.steps:
            results.append({
                "technique": step.technique,
                "description": step.description,
                "success": step.success,
                "duration": step.duration,
                "params": step.params,
                "improvement_score": step.improvement_score
            })
        return results

    def get_initial_analysis(self) -> Dict:
        """Get initial image analysis results"""
        return self.initial_analysis

    def get_final_analysis(self) -> Dict:
        """Get final image analysis results"""
        return self.final_analysis

    def save_intermediate_results(self, folder_path: str):
        """Save intermediate results to a folder"""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for i, step in enumerate(self.steps):
            filename = os.path.join(folder_path,
                                    f"{i:02d}_{step.technique}.png")
            cv2.imwrite(filename, step.output_image)
            logger.info(f"Saved intermediate result: {filename}")

    def save_final_result(self, file_path: str):
        """Save the final optimized image to file"""
        cv2.imwrite(file_path, self.optimized_image)
        logger.info(f"Saved final optimized image: {file_path}")

    def reset(self):
        """Reset the optimizer to initial state"""
        self.original_image = None
        self.current_image = None
        self.optimized_image = None
        self.gray_image = None
        self.steps = []
        self.initial_analysis = None
        self.final_analysis = None
        logger.info("Image optimizer reset")

    def list_profiles(self) -> List[str]:
        """List available optimization profiles"""
        return self.profile_manager.list_profiles()


# if __name__ == "__main__":
#     # Example usage
#     optimizer = ImageOptimizer()
#     optimizer.load_image("test_image.jpg")
#     optimizer.optimize()
#     optimizer.save_final_result("optimized_image.jpg")
#     results = optimizer.get_step_results()
#     for result in results:
#         print(result)
#     final_analysis = optimizer.get_final_analysis()
#     print(final_analysis)
