"""
This module contains line detection algorithms.

Currently, two algorithms are implemented:
    - HoughLineDetector: uses Canny edge detection and Hough transform
    - LineSegmentDetector: uses the LineSegmentDetector method of openCV

These detectors can be used to extract lines from an image, which can then be used to estimate
    the vanishing point or correct perspective distortion.
"""

import os
from abc import ABC, abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

DEFAULT_CONFIG_FOLDER: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"config")

DEFAULT_CONFIG_HOUGH_LINE: str = os.path.join(DEFAULT_CONFIG_FOLDER, r"hough_line_parameters.yaml")
DEFAULT_CONFIG_LSD: str = os.path.join(DEFAULT_CONFIG_FOLDER, r"lsd_parameters.yaml")

DEFAULT_GAUSSIAN_KERNEL_SIZE: int = 5
DEFAULT_CANNY_APERTURE_SIZE: int = 5
DEFAULT_CANNY_THRESHOLD_1: int = 0
DEFAULT_CANNY_THRESHOLD_2: int = 30
DEFAULT_HOUGH_RHO: int = 1
DEFAULT_HOUGH_THETA: float = np.pi / 180.0
DEFAULT_HOUGH_THRESHOLD: int = 50
DEFAULT_HOUGH_MIN_LINE_LENGTH: int = 75
DEFAULT_HOUGH_MAX_LINE_GAP: int = 10


class CommonLineDetector(ABC):
    """
    Abstract base class for line detectors.
    """
    def __init__(self, config_file: str =None) -> None:
        """
        Initialize the CommonLineDetector.

        Args:
            config_file (str, optional): Path to the configuration file. Defaults to None.
        """
        self.config = {}
        if (
            config_file is not None
            and os.path.splitext(config_file)[-1].lower() == ".yaml"
            and os.path.exists(config_file)
        ):
            with open(config_file, "r", encoding="utf-8") as yamlfile:
                self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    @abstractmethod
    def detect_lines(self, img: np.ndarray) -> np.ndarray:
        """
        Detect lines in an image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Detected lines as an array of line parameters.
        """

    @staticmethod
    def visualize_lines(img: np.ndarray, lines: np.ndarray) -> None:
        """
        Visualize detected lines on an image.

        Args:
            img (np.ndarray): Input image.
            lines (np.ndarray): Array of line parameters.
        """
        display_img = img.copy()
        for line in lines:
            cv2.line(display_img, line[:2], line[2:], (0,255,0), 1)
        if display_img.shape[2] > 1:
            display_img = display_img[:,:,::-1]

        plt.imshow(display_img)
        plt.show(block=True)

    @staticmethod
    def keep_vertical_lines(lines: np.ndarray, acceptable_angle_offset: float =15) -> np.ndarray:
        """
        Filter and keep only the vertical lines from an array of line parameters.

        Args:
            lines (np.ndarray): Array of line parameters.
            acceptable_angle_offset (float, optional):
                Acceptable angle offset in degrees for vertical.
                Defaults to 15.

        Returns:
            np.ndarray: Filtered array of line parameters containing only vertical lines.
        """
        vertical_lines = np.array([])
        for line in lines:
            x1, y1, x2, y2 = line
            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if (
                abs(angle) > (90 - acceptable_angle_offset)
                and abs(angle) < (90 + acceptable_angle_offset)
            ):
                vertical_lines = (np.vstack([vertical_lines, [x1, y1, x2, y2]])
                                  if vertical_lines.size else np.array([x1, y1, x2, y2]))
        return vertical_lines

class HoughLineDetector(CommonLineDetector):
    """
    Line detector using Hough line transform.
    """
    def __init__(self, config_file: str=DEFAULT_CONFIG_HOUGH_LINE) -> None:
        """
        Initialize the HoughLineDetector.

        Args:
            config_file (str, optional): Path to the configuration file.
                Defaults to DEFAULT_CONFIG_HOUGH_LINE.
        """
        super().__init__(config_file)
        # Read parameters from config file
        self.gaussian_kernel_size = self.config.get("gaussian_kernel_size",
                                                    DEFAULT_GAUSSIAN_KERNEL_SIZE)
        self.canny_aperture_size = self.config.get("canny_aperture_size",
                                                   DEFAULT_CANNY_APERTURE_SIZE)
        self.canny_threshold1 = self.config.get("canny_threshold1", DEFAULT_CANNY_THRESHOLD_1)
        self.canny_threshold2 = self.config.get("canny_threshold2", DEFAULT_CANNY_THRESHOLD_2)
        self.hough_rho = self.config.get("hough_rho", DEFAULT_HOUGH_RHO)
        self.hough_theta = self.config.get("hough_theta", DEFAULT_HOUGH_THETA)
        self.hough_threshold = self.config.get("hough_threshold", DEFAULT_HOUGH_THRESHOLD)
        self.hough_min_line_length = self.config.get("hough_min_line_length",
                                                     DEFAULT_HOUGH_MIN_LINE_LENGTH)
        self.hough_max_line_gap = self.config.get("hough_max_line_gap", DEFAULT_HOUGH_MAX_LINE_GAP)

    def detect_lines(self, img: np.ndarray) -> np.ndarray:
        """
        Detect lines in an image using Hough line transform.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Detected lines as an array of line parameters.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
        edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2,
                          self.canny_aperture_size)
        hough_lines = cv2.HoughLinesP(
            edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        hough_lines = hough_lines[:, 0]
        lines = self.keep_vertical_lines(hough_lines)

        return lines


class LineSegmentDetector(CommonLineDetector):
    """
    Line detector using Line Segment Detector method of OpenCV.
    """
    def __init__(self, config_file: str=DEFAULT_CONFIG_LSD) -> None:
        """
        Initialize the LineSegmentDetector.

        Args:
            config_file (str, optional): Path to the configuration file.
                Defaults to DEFAULT_CONFIG_LSD.
        """
        super().__init__(config_file)
        self.gaussian_kernel_size = self.config.get("gaussian_kernel_size",
                                                    DEFAULT_GAUSSIAN_KERNEL_SIZE)

    def detect_lines(self, img: np.ndarray) -> np.ndarray:
        """
        Detect lines in an image using Line Segment Detector (LSD) method.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Detected lines as an array of line parameters.
        """
        lsd = cv2.createLineSegmentDetector(0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)

        lsd_lines = lsd.detect(gray)[0]
        lsd_lines = lsd_lines[:, 0]

        lines = self.keep_vertical_lines(lsd_lines)

        return lines
