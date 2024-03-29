"""
This module contains a perspective corrector.
The perspective corrector takes an image and the coordinates of the vanishing point as input,
    and outputs a corrected image.
The correction is done by computing the homography matrix that maps the distorted image
    to a rectified image.
"""

from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.line_detector import HoughLineDetector, LineSegmentDetector
from src.vanishing_point_detector import VanishingPointDetector

SUPPORTED_LINE_DETECTOR: list[str] = ["HoughLineDetector", "LineSegmentDetector"]

class PerspectiveCorrector:
    """
    A class to correct the perspective of an image using vanishing points.
    """
    def __init__(self, line_detector="LineSegmentDetector", config_line_detector_file=None):
        """
        Initializes a PerspectiveCorrector object.
        Args:
            line_detector (str): The name of the line detector to be used.
                Defaults to "LineSegmentDetector".
            config_line_detector_file (str): Path to a configuration file for the line detector.
                Defaults to None.
        """
        if line_detector not in SUPPORTED_LINE_DETECTOR:
            raise NotImplementedError(f"The line detector {line_detector} is not supported "
                                      f"The supported line detectors are {SUPPORTED_LINE_DETECTOR}")
        if line_detector == "LineSegmentDetector":
            self.line_detector = LineSegmentDetector(config_file=config_line_detector_file)
        elif line_detector == "HoughLineDetector":
            self.line_detector = HoughLineDetector(config_file=config_line_detector_file)

    def run(self, img, visualize_lines=False, display_corrected_images=False
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Corrects the perspective of the input image using vanishing points and returns
            the corrected image.
        Args:
            img (numpy.ndarray): The input image to be corrected.
            visualize_lines (bool): Whether to display the detected lines on the input image.
                Defaults to False.
            display_corrected_images (bool): Whether to display the corrected image.
                Defaults to False.
        Returns:
            numpy.ndarray: The corrected image.
        """
        # First step detects the line on the image
        lines = self.line_detector.detect_lines(img)
        if visualize_lines:
            self.line_detector.visualize_lines(img, lines)

        # Detect the vanishing point on the image
        vanishing_point_detector = VanishingPointDetector()
        vanishing_vertical_point = vanishing_point_detector.detect_vanishing_point(lines)

        vanishing_horizontal_point = np.array([1, 0, 0])
        height, width, _ = img.shape

        homography = self.find_homography(vanishing_horizontal_point, vanishing_vertical_point)
        corrected_image = cv2.warpPerspective(img, homography, (width, height))

        corners = np.array([
            [0, 0, width, width],
            [0, height, 0, height],
            [1, 1, 1, 1]]
        )

        transformed_corners = homography @ corners
        transformed_corners = transformed_corners[:2] / transformed_corners[2]
        x_crop_min = int(np.max([transformed_corners[0,0], transformed_corners[0, 1]]))
        x_crop_max = int(np.min([transformed_corners[0,2], transformed_corners[0, 3]]))

        y_crop_min = int(np.max([transformed_corners[1,0], transformed_corners[1, 2]]))
        y_crop_max = int(np.min([transformed_corners[1,1], transformed_corners[1, 3]]))

        # TODO: find a better way to handle with incorrect transformation
        if x_crop_min >= x_crop_max:
            x_crop_min = 0
            x_crop_max = width - 1
        if y_crop_min >= y_crop_max:
            y_crop_min = 0
            y_crop_max = height - 1

        crop_image = cv2.resize(
            corrected_image[y_crop_min:y_crop_max, x_crop_min:x_crop_max],
            (width, height),
            interpolation=cv2.INTER_CUBIC
        )

        if display_corrected_images:
            _, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
            # Display the image in the first column
            axs[0].imshow(img[:,:,::-1])
            axs[0].axis("off")
            axs[0].set_title("Image")

            # Display the image in the first column
            axs[1].imshow(corrected_image[:,:,::-1])
            axs[1].axis("off")
            axs[1].set_title("Corrected Image")

            # Display the image in the first column
            axs[2].imshow(crop_image[:,:,::-1])
            axs[2].axis("off")
            axs[2].set_title("Crop Image")
            plt.show(block=True)

        return corrected_image, crop_image


    @staticmethod
    def find_homography(vp1: np.ndarray, vp2: np.ndarray) -> np.ndarray:
        """
        Computes the homography matrix for transforming an image using vanishing points.

        Args:
            vp1 (np.ndarray): 3D numpy array representing the first vanishing point.
            vp2 (np.ndarray): 3D numpy array representing the second vanishing point.

        Returns:
            np.ndarray: The homography matrix.
        """
        vanishing_line = np.cross(vp1, vp2)
        H = np.eye(3)
        H[2] = vanishing_line / vanishing_line[2]
        H = H / H[2, 2]

        # Find vertical vanishing point after homography
        v_post2 = H @ vp2
        v_post2 = v_post2 / np.sqrt(v_post2[0]**2 + v_post2[1]**2)
        # Find angle between vertical vanishing point and y axis [0, 1, 0]
        theta = np.arccos(np.dot(v_post2, [0, 1, 0]) / np.sqrt(v_post2[0]**2 + v_post2[1]**2))
        # Rotate y axis
        R = np.array([[1, np.sin(theta), 0],
            [0, np.cos(theta), 0],
            [0, 0, 1]])
        # if reflection,
        if np.linalg.det(R) < 0:
            R[:, 1] = -R[:, 1]

        R = np.linalg.inv(R)
        homography = R @ H
        return homography
