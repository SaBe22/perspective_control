import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple

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
            line_detector (str): The name of the line detector to be used. Defaults to "LineSegmentDetector".
            config_line_detector_file (str): Path to a configuration file to be used by the line detector.
                Defaults to None.
        """
        if line_detector not in SUPPORTED_LINE_DETECTOR:
            raise NotImplementedError(f"The line detector {line_detector} is not supported "
                                      f"The supported line detectors are {SUPPORTED_LINE_DETECTOR}")
        if line_detector == "LineSegmentDetector":
            self.line_detector = LineSegmentDetector(config_file=config_line_detector_file)
        elif line_detector == "HoughLineDetector":
            self.line_detector = HoughLineDetector(config_file=config_line_detector_file)

    def run(self, img, visualize_lines=False, display_corrected_images=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Corrects the perspective of the input image using vanishing points and returns the corrected image.

        Args:
            img (numpy.ndarray): The input image to be corrected.
            visualize_lines (bool): Whether to display the detected lines on the input image. Defaults to False.
            display_corrected_images (bool): Whether to display the corrected image. Defaults to False.

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

        homography = self.find_homography(vanishing_horizontal_point, vanishing_vertical_point, width, height)
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

        crop_image = cv2.resize(corrected_image[y_crop_min:y_crop_max, x_crop_min:x_crop_max], (width, height))

        if display_corrected_images:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
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

        return corrected_image, crop_image


    @staticmethod
    def find_homography(vp1: np.ndarray, vp2: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Computes the homography matrix for transforming an image using vanishing points.

        Args:
            vp1 (np.ndarray): 3D numpy array representing the first vanishing point.
            vp2 (np.ndarray): 3D numpy array representing the second vanishing point.
            width (int): Width of the input image.
            height (int): Height of the input image.

        Returns:
            np.ndarray: The homography matrix.
        """
        vanishing_line = np.cross(vp1, vp2)
        H = np.eye(3)
        H[2] = vanishing_line / vanishing_line[2]
        H = H / H[2, 2]

        # Find directions corresponding to vanishing points
        v_post1 = H @ vp1
        v_post2 = H @ vp2
        v_post1 = v_post1 / np.sqrt(v_post1[0]**2 + v_post1[1]**2)
        v_post2 = v_post2 / np.sqrt(v_post2[0]**2 + v_post2[1]**2)

        directions = np.array([[v_post1[0], -v_post1[0], v_post2[0], -v_post2[0]],
                            [v_post1[1], -v_post1[1], v_post2[1], -v_post2[1]]])

        thetas = np.arctan2(directions[0], directions[1])

        # Find direction closest to horizontal axis
        h_ind = np.argmin(np.abs(thetas))

        # Find positive angle among the rest for the vertical axis
        if h_ind // 2 == 0:
            v_ind = 2 + np.argmax([thetas[2], thetas[3]])
        else:
            v_ind = np.argmax([thetas[2], thetas[3]])

        A1 = np.array([[directions[0, v_ind], directions[0, h_ind], 0],
                    [directions[1, v_ind], directions[1, h_ind], 0],
                    [0, 0, 1]])
        # Check for reflection and remove if necessary
        if np.linalg.det(A1) < 0:
            A1[:, 0] = -A1[:, 0]

        A = np.linalg.inv(A1)

        homography = A @ H

        return homography
