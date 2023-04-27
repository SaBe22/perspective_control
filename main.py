"""
This module contains the main script for running the perspective correction pipeline.

The script takes an input image and saves the output corrected image to a file.
The user can also choose to display the detected
lines and/or the corrected image on the screen.

Example usage:
python main.py -i input_image.png -o output_image.png --display_lines --display_results
"""

import argparse
import os

import cv2

from src.perspective_corrector import PerspectiveCorrector

SUPPORTED_IMAGES_EXT = [".jpg", ".jpeg", ".png", ".tiff"]

def main() -> None:
    """
    Parse command line arguments, initialize the model, parameters, and hyperparameters,
    train the model, and save the best model.
    """
    args = parse_args()
    input_image_path = args.input_image
    output_image_path = args.output_image
    display_lines = args.display_lines
    display_results = args.display_results

    # Check that the input image is an image
    if (
        not os.path.exists(input_image_path)
        or not os.path.splitext(input_image_path)[-1].lower() in SUPPORTED_IMAGES_EXT
    ):
        raise ValueError(f"Input image given {input_image_path} must be"
                         " in the following format {SUPPORTED_IMAGES_EXT}")

    img = cv2.imread(input_image_path)

    perspective_corrector = PerspectiveCorrector()
    _, crop_image = perspective_corrector.run(img, visualize_lines=display_lines,
                                              display_corrected_images=display_results)

    if not os.path.splitext(output_image_path)[-1].lower() in SUPPORTED_IMAGES_EXT:
        raise ValueError(f"The provided output image path {output_image_path} must be"
                         " in the following format {SUPPORTED_IMAGES_EXT}")

    # Create output folder if it doesn't exist
    output_folder = os.path.dirname(output_image_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the results
    cv2.imwrite(output_image_path, crop_image)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments and return the parsed arguments as a Namespace object.
    Returns:
        argparse.Namespace: Object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Perspective Correction")

    parser.add_argument("-i", "--input_image", required=True, help="Input image file path")
    parser.add_argument("-o", "--output_image", required=True, help="Output image file path")
    parser.add_argument("--display_lines", action="store_true", help="Display detected lines")
    parser.add_argument("--display_results", action="store_true", help="Display corrected images")

    return parser.parse_args()

if __name__ == "__main__":
    main()
