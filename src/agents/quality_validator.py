import os
from promptflow.core import tool

from tools.quality_validator import ImageQualityAnalyzer


@tool
async def quality_validator(folder_path: str):
    files = os.listdir(folder_path)
    summaries = []
    for file in files:
        file_path = os.path.join(folder_path, file)
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

            summary = {
                "file_name": file,
                "overall_score": results['overall_score'],
                "overall_threshold": results['overall_threshold'],
                "ocr_ready": 'Yes' if results['passed'] else 'No',
                "metrics": results["metrics"]
            }
            summaries.append(summary)

            # Get preprocessing recommendations
            recommendations = analyzer.get_preprocessing_recommendations()
            print("\nRecommendations:")
            for key, recommendation in recommendations.items():
                print(f"- {key.title()}: {recommendation}")
        else:
            print(f"Failed to load image: {file_path}")

    return summaries
