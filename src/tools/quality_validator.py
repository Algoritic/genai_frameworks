import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from core.logger import logger

# Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """Class to store quality metric information with thresholds"""
    name: str
    value: float = 0.0
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    weight: float = 1.0
    passed: bool = False

    def evaluate(self) -> bool:
        """Evaluate if the metric passes the threshold"""
        self.passed = self.min_threshold <= self.value <= self.max_threshold
        return self.passed

    def get_score(self) -> float:
        """Calculate weighted score for this metric"""

        if self.passed:
            if (self.value == self.max_threshold):
                return 1.0 * self.weight
            if 0.0 <= self.value <= 1.0:
                normalized = self.value  # Use value directly if within 0-1
            elif self.max_threshold == self.min_threshold:
                normalized = 1.0
            else:
                # Scale only if the value is outside the [0,1] range
                normalized = (self.value - self.min_threshold) / (
                    self.max_threshold - self.min_threshold)
                normalized = max(0.0, min(1.0,
                                          normalized))  # Clamp within [0,1]

            # Convert to 0-1 where values closer to center of range are better
            normalized = 1.0 - abs(2 * normalized - 1.0)
            return normalized * self.weight
        return 0.0


class QualityPreset:
    """Class to define presets for different document types and scenarios"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.metrics = {}
        self.overall_threshold = 60.0  # Overall score threshold (0-100)

    def add_metric(self, name: str, min_threshold: float, max_threshold: float,
                   weight: float):
        """Add a metric with its thresholds to the preset"""
        self.metrics[name] = {
            'min_threshold': min_threshold,
            'max_threshold': max_threshold,
            'weight': weight
        }

    def get_metric_config(self, name: str) -> Dict:
        """Get configuration for a specific metric"""
        return self.metrics.get(name, {
            'min_threshold': 0.0,
            'max_threshold': 1.0,
            'weight': 1.0
        })


class PresetManager:
    """Class to manage different quality presets"""

    def __init__(self):
        self.presets = {}
        self._initialize_default_presets()

    def _initialize_default_presets(self):
        """Initialize default presets for common scenarios"""
        # Standard printed documents
        standard = QualityPreset(
            "standard", "Standard printed documents like books or reports")
        standard.add_metric("contrast", 0.4, 1.0, 1.5)
        standard.add_metric("brightness", 0.4, 0.6, 1.0)
        standard.add_metric("sharpness", 0.5, 1.0, 1.5)
        standard.add_metric("noise", 0.0, 0.3, 1.0)
        standard.add_metric("skew_angle", 0.0, 5.0, 1.0)
        standard.add_metric("dpi", 200, 600, 1.0)
        standard.overall_threshold = 70.0
        self.presets["standard"] = standard

        # Low-quality documents
        low_quality = QualityPreset(
            "low_quality", "Receipts, faxes, or thermal paper prints")
        low_quality.add_metric("contrast", 0.3, 1.0, 1.5)
        # low_quality.add_metric("brightness", 0.3, 0.7, 1.0)
        low_quality.add_metric("sharpness", 0.3, 1.0, 1.0)
        low_quality.add_metric("noise", 0.0, 0.5, 1.0)
        # low_quality.add_metric("skew_angle", 0.0, 10.0, 0.8)
        low_quality.add_metric("dpi", 150, 600, 1.0)
        low_quality.overall_threshold = 50.0
        self.presets["low_quality"] = low_quality

        # Handwritten documents
        handwritten = QualityPreset("handwritten",
                                    "Handwritten notes and documents")
        handwritten.add_metric("contrast", 0.4, 1.0, 1.5)
        handwritten.add_metric("brightness", 0.4, 0.7, 1.0)
        handwritten.add_metric("sharpness", 0.6, 1.0, 1.5)
        handwritten.add_metric("noise", 0.0, 0.3, 1.0)
        handwritten.add_metric("skew_angle", 0.0, 15.0, 0.5)
        handwritten.add_metric("dpi", 300, 600, 1.5)
        handwritten.overall_threshold = 65.0
        self.presets["handwritten"] = handwritten

        # Smartphone photo
        smartphone = QualityPreset(
            "smartphone", "Photos of documents taken with smartphone")
        smartphone.add_metric("contrast", 0.3, 1.0, 1.5)
        smartphone.add_metric("brightness", 0.3, 0.7, 1.0)
        smartphone.add_metric("sharpness", 0.4, 1.0, 1.5)
        smartphone.add_metric("noise", 0.0, 0.4, 1.0)
        smartphone.add_metric("skew_angle", 0.0, 20.0, 0.5)
        smartphone.add_metric("dpi", 150, 600, 1.0)
        smartphone.overall_threshold = 55.0
        self.presets["smartphone"] = smartphone

    def get_preset(self, name: str = "standard") -> QualityPreset:
        """Get a preset by name, defaults to standard if not found"""
        return self.presets.get(name, self.presets["standard"])

    def list_presets(self) -> List[str]:
        """List all available presets"""
        return list(self.presets.keys())

    def add_preset(self, preset: QualityPreset):
        """Add a new preset"""
        self.presets[preset.name] = preset


class ImageQualityAnalyzer:
    """Main class to analyze image quality for OCR"""

    def __init__(self, preset_name: str = "standard"):
        self.preset_manager = PresetManager()
        self.preset = self.preset_manager.get_preset(preset_name)
        self.metrics = {}
        self.image = None
        self.gray_image = None
        self.height = 0
        self.width = 0
        self.overall_score = 0.0
        self.passed = False

    def set_preset(self, preset_name: str):
        """Change the current preset"""
        self.preset = self.preset_manager.get_preset(preset_name)
        logger.info(f"Changed preset to: {preset_name}")

    def load_image(self, image_path: str) -> bool:
        """Load an image from file"""
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return False

        try:
            self.image = cv2.imread(image_path)
            if self.image is None:
                logger.error(f"Failed to read image: {image_path}")
                return False

            self.height, self.width = self.image.shape[:2]
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            logger.info(
                f"Successfully loaded image: {image_path} ({self.width}x{self.height})"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return False

    def _calculate_contrast(self) -> float:
        """Calculate image contrast using standard deviation"""
        if self.gray_image is None:
            return 0.0

        std_dev = np.std(self.gray_image.astype(np.float32))
        # Normalize to 0-1 range (typical std dev range is 0-127)
        normalized = min(std_dev / 127.0, 1.0)
        return normalized

    def _calculate_brightness(self) -> float:
        """Calculate average brightness of the image (0-1)"""
        if self.gray_image is None:
            return 0.0

        return np.mean(self.gray_image) / 255.0

    def _calculate_sharpness(self) -> float:
        """Estimate image sharpness using Laplacian variance"""
        if self.gray_image is None:
            return 0.0

        # Apply Laplacian filter
        laplacian = cv2.Laplacian(self.gray_image, cv2.CV_64F)
        # Calculate variance
        variance = np.var(laplacian)
        # Normalize (typical range 0-1000)
        normalized = min(variance / 1000.0, 1.0)
        return normalized

    def _calculate_noise(self) -> float:
        """Estimate image noise level"""
        if self.gray_image is None:
            return 0.0

        # Apply median blur to remove noise
        denoised = cv2.medianBlur(self.gray_image, 5)
        # Calculate difference between original and denoised
        diff = cv2.absdiff(self.gray_image, denoised)
        # Calculate normalized noise level
        noise_level = np.mean(diff) / 255.0
        return noise_level

    def _calculate_skew_angle(self) -> float:
        """Estimate text skew angle in degrees."""
        if self.gray_image is None:
            return 0.0

        # Apply Otsu's threshold to binarize the image
        _, thresh = cv2.threshold(self.gray_image, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours and compute angles
        angles = np.array([(angle + 90 if angle < -45 else angle -
                            90) if angle > 45 else angle
                           for _, _, angle in (cv2.minAreaRect(cnt)
                                               for cnt in contours
                                               if cv2.contourArea(cnt) >= 50)])

        # Return average absolute angle or 0 if no valid contours
        return np.mean(np.abs(angles)) if angles.size > 0 else 0.0

    def _estimate_dpi(self) -> float:
        """Estimate image DPI based on size"""
        if self.image is None:
            return 0.0

        # This is a rough estimation - in reality DPI depends on physical size
        # Assumption: typical letter/A4 document is around 8.5x11 inches
        # So a good scan would be around 2550x3300 pixels at 300 DPI

        area = self.width * self.height
        estimated_dpi = (area / (8.5 * 11))**0.5
        return estimated_dpi

    def analyze(self) -> Dict:
        """Perform all quality checks and return results"""
        if self.image is None:
            logger.error("No image loaded for analysis")
            return {"error": "No image loaded"}

        # Calculate metrics
        contrast = self._calculate_contrast()
        # brightness = self._calculate_brightness()
        sharpness = self._calculate_sharpness()
        noise = self._calculate_noise()
        # skew_angle = self._calculate_skew_angle()
        dpi = self._estimate_dpi()

        # Create metric objects with preset thresholds
        self.metrics = {
            "contrast":
            QualityMetric(name="contrast",
                          value=contrast,
                          **self.preset.get_metric_config("contrast")),
            # "brightness":
            # QualityMetric(name="brightness",
            #               value=brightness,
            #               **self.preset.get_metric_config("brightness")),
            "sharpness":
            QualityMetric(name="sharpness",
                          value=sharpness,
                          **self.preset.get_metric_config("sharpness")),
            "noise":
            QualityMetric(name="noise",
                          value=noise,
                          **self.preset.get_metric_config("noise")),
            # "skew_angle":
            # QualityMetric(name="skew_angle",
            #               value=skew_angle,
            #               **self.preset.get_metric_config("skew_angle")),
            "dpi":
            QualityMetric(name="dpi",
                          value=dpi,
                          **self.preset.get_metric_config("dpi"))
        }

        # Evaluate each metric
        for metric in self.metrics.values():
            metric.evaluate()

        # Calculate overall score
        total_weight = sum(metric.weight for metric in self.metrics.values())
        weighted_sum = sum(metric.get_score() * 100
                           for metric in self.metrics.values())

        if total_weight > 0:
            self.overall_score = weighted_sum / total_weight
        else:
            self.overall_score = 0

        # Check if image passes overall threshold
        self.passed = self.overall_score >= self.preset.overall_threshold

        # Prepare results
        results = {
            "preset_used": self.preset.name,
            "overall_score": self.overall_score,
            "overall_threshold": self.preset.overall_threshold,
            "passed": self.passed,
            "metrics": {
                name: {
                    "value": metric.value,
                    "passed": metric.passed,
                    "min_threshold": metric.min_threshold,
                    "max_threshold": metric.max_threshold,
                    "weight": metric.weight,
                    "score": metric.get_score() * 100
                }
                for name, metric in self.metrics.items()
            },
            "image_info": {
                "width": self.width,
                "height": self.height,
                "channels":
                self.image.shape[2] if len(self.image.shape) > 2 else 1
            }
        }

        logger.info(
            f"Analysis complete - Overall score: {self.overall_score:.1f}/100 (Threshold: {self.preset.overall_threshold}) - Passed: {self.passed}"
        )
        return results

    def visualize_results(self, save_path: Optional[str] = None):
        """Visualize analysis results"""
        if not self.metrics:
            logger.error("No analysis results to visualize")
            return

        # Create figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(
            f'Image Quality Analysis - {self.preset.name.title()} Preset',
            fontsize=16)

        # Plot original image
        axs[0, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')

        # Plot grayscale image
        axs[0, 1].imshow(self.gray_image, cmap='gray')
        axs[0, 1].set_title('Grayscale Image')
        axs[0, 1].axis('off')

        # Plot contrast visualization
        edges = cv2.Canny(self.gray_image, 100, 200)
        axs[1, 0].imshow(edges, cmap='gray')
        axs[1, 0].set_title(
            f'Edges (Contrast: {self.metrics["contrast"].value:.2f})')
        axs[1, 0].axis('off')

        # Plot sharpness visualization
        laplacian = cv2.Laplacian(self.gray_image, cv2.CV_64F)
        laplacian_norm = cv2.normalize(laplacian, None, 0, 255,
                                       cv2.NORM_MINMAX).astype(np.uint8)
        axs[1, 1].imshow(laplacian_norm, cmap='hot')
        axs[1, 1].set_title(
            f'Laplacian (Sharpness: {self.metrics["sharpness"].value:.2f})')
        axs[1, 1].axis('off')

        # Create bar chart for metric scores
        metric_names = list(self.metrics.keys())
        metric_scores = [
            self.metrics[name].get_score() * 100 for name in metric_names
        ]
        metric_colors = [
            'green' if self.metrics[name].passed else 'red'
            for name in metric_names
        ]

        axs[2, 0].bar(metric_names, metric_scores, color=metric_colors)
        axs[2, 0].set_title('Metric Scores')
        axs[2, 0].set_ylim(0, 100)
        axs[2, 0].set_ylabel('Score')
        axs[2, 0].tick_params(axis='x', rotation=45)

        # Plot overall score gauge
        gauge_ax = axs[2, 1]
        gauge_ax.set_title(f'Overall Score: {self.overall_score:.1f}/100')
        gauge_ax.axis('equal')

        # Create a simple gauge chart
        wedgeprops = {'width': 0.3, 'edgecolor': 'black', 'linewidth': 1}

        # Background gauge (gray)
        gauge_ax.pie([1],
                     startangle=90,
                     counterclock=False,
                     wedgeprops=dict(wedgeprops, edgecolor='gray'),
                     radius=1.0,
                     colors=['lightgray'])

        # Score gauge (colored based on pass/fail)
        score_color = 'green' if self.passed else 'red'
        score_angle = self.overall_score * 3.6  # Convert to degrees (0-360)
        gauge_ax.pie([score_angle, 360 - score_angle],
                     startangle=90,
                     counterclock=False,
                     wedgeprops=wedgeprops,
                     radius=1.0,
                     colors=[score_color, 'white'])

        # Add threshold marker
        threshold_angle = self.preset.overall_threshold * 3.6
        threshold_radian = np.radians(90 - threshold_angle)
        x = np.cos(threshold_radian)
        y = np.sin(threshold_radian)
        gauge_ax.plot([0, x], [0, y], 'b-', linewidth=2)
        gauge_ax.text(x * 1.1,
                      y * 1.1,
                      f"Threshold\n{self.preset.overall_threshold}%",
                      ha='center',
                      va='center',
                      color='blue')

        # Add center text
        gauge_ax.text(0,
                      0,
                      "OCR\nready" if self.passed else "Not\nready",
                      ha='center',
                      va='center',
                      fontsize=12,
                      fontweight='bold')

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")

        plt.show()

    def get_preprocessing_recommendations(self) -> Dict[str, str]:
        """Get recommendations for preprocessing based on analysis results"""
        if not self.metrics:
            return {"error": "No analysis results available"}

        recommendations = {}

        # Contrast recommendations
        if not self.metrics["contrast"].passed:
            if self.metrics["contrast"].value < self.metrics[
                    "contrast"].min_threshold:
                recommendations[
                    "contrast"] = "Increase image contrast using histogram equalization"
            else:
                recommendations["contrast"] = "Reduce excessive contrast"

        # # Brightness recommendations
        # if not self.metrics["brightness"].passed:
        #     if self.metrics["brightness"].value < self.metrics[
        #             "brightness"].min_threshold:
        #         recommendations["brightness"] = "Increase image brightness"
        #     else:
        #         recommendations["brightness"] = "Reduce image brightness"

        # Sharpness recommendations
        if not self.metrics["sharpness"].passed:
            if self.metrics["sharpness"].value < self.metrics[
                    "sharpness"].min_threshold:
                recommendations[
                    "sharpness"] = "Apply sharpening filter to improve text edges"

        # Noise recommendations
        if not self.metrics["noise"].passed:
            if self.metrics["noise"].value > self.metrics[
                    "noise"].max_threshold:
                recommendations[
                    "noise"] = "Apply noise reduction filter (median or bilateral filter)"

        # Skew recommendations
        # if not self.metrics["skew_angle"].passed:
        #     if self.metrics["skew_angle"].value > self.metrics[
        #             "skew_angle"].max_threshold:
        #         recommendations[
        #             "skew"] = f"Deskew image (current angle: {self.metrics['skew_angle'].value:.1f} degrees)"

        # DPI recommendations
        if not self.metrics["dpi"].passed:
            if self.metrics["dpi"].value < self.metrics["dpi"].min_threshold:
                recommendations[
                    "dpi"] = f"Rescan at higher resolution (current est. DPI: {self.metrics['dpi'].value:.0f})"

        # If all passed, suggest proceeding with OCR
        if not recommendations:
            recommendations["status"] = "Image quality is suitable for OCR"

        return recommendations


def main():
    """Example usage of the OCR Image Quality Analyzer"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate image quality for OCR")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--preset",
                        default="standard",
                        help="Quality preset to use")
    parser.add_argument("--visualize",
                        action="store_true",
                        help="Show visualization")
    parser.add_argument("--save-viz", help="Save visualization to file")
    args = parser.parse_args()

    # Initialize analyzer with selected preset
    analyzer = ImageQualityAnalyzer(args.preset)

    # List available presets
    print(
        f"Available presets: {', '.join(analyzer.preset_manager.list_presets())}"
    )
    print(f"Using preset: {args.preset}")

    # Load and analyze image
    if analyzer.load_image(args.image_path):
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

        # Show visualization if requested
        if args.visualize or args.save_viz:
            analyzer.visualize_results(args.save_viz)
    else:
        print(f"Failed to load image: {args.image_path}")


if __name__ == "__main__":
    main()
