"""
This module contains a vanishing point detector.

The vanishing point detector takes a list of lines as input,
and outputs the coordinates of the vanishing point.
The current implementation uses a voting scheme to estimate the vanishing point from the lines.
"""

from dataclasses import dataclass

import numpy as np

SEED: int = 42
MAX_ITER: int = 1000

@dataclass
class EdgesDescriptor:
    """
    Data class to represent edge descriptors for vanishing point detection.

    Attributes:
    -----------
    pt1 : numpy.ndarray
        Array of shape (N, 3) representaing the locations of the N
    pt1 : numpy.ndarray
        Array of shape (N, 3) representaing the locations of the N
    edges_locations : numpy.ndarray
        Array of shape (N, 3) representing the locations of N edges in the image.
    edges_directions : numpy.ndarray
        Array of shape (N, 3) representing the directions of N edges in the image.
    edges_norm : numpy.ndarray
        Array of shape (N,) representing the norm (length) of N edges in the image.
    lines_homogeneous_eqs : numpy.ndarray
        Array of shape (N, 3) representing the homogeneous equations of N lines in the image.
    num_lines : int
        Total number of lines in the image.
    """
    num_lines: int
    pt1: np.ndarray
    pt2: np.ndarray
    lines_homogeneous_eqs: np.ndarray
    edges_norm: np.ndarray
    edges_locations: np.ndarray
    edges_directions: np.ndarray

class VanishingPointDetector:
    """
    Class for vanishing point detection.

    Attributes:
    -----------
    max_iter : int
        Maximum number of iterations for RANSAC.
    theta_thresh : float
    """
    def __init__(self, max_iter: int =MAX_ITER, theta_thresh: float =5) -> None:
        """
        Initialize HomographyEstimator class.

        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations for RANSAC.
        """
        self.max_iter = int(max_iter)
        self.theta_thresh = theta_thresh * np.pi / 180
        np.random.seed(SEED)

    @staticmethod
    def preprocess_lines(lines: np.ndarray) -> EdgesDescriptor:
        """
        Preprocess the lines for vanishing point detection.

        This method normalizes the edge directions and computes the homogeneous equations
        of the lines from edge descriptors.
        """
        num_lines = lines.shape[0]
        pt1 = np.column_stack((lines[:, :2], np.ones(num_lines, dtype=np.float32)))
        pt2 = np.column_stack((lines[:, 2:], np.ones(num_lines, dtype=np.float32)))
        lines_homogeneous_eqs = np.cross(pt1, pt2)
        edges_norm = np.linalg.norm(pt2 - pt1, axis=1)
        edges_directions = (pt2 - pt1) / np.tile(edges_norm.reshape(-1, 1), (1, 3))
        edges_locations = (pt1 + pt2) / 2

        return EdgesDescriptor(
            num_lines = num_lines,
            pt1 = pt1,
            pt2 = pt2,
            lines_homogeneous_eqs = lines_homogeneous_eqs,
            edges_norm = edges_norm,
            edges_directions = edges_directions,
            edges_locations = edges_locations
        )

    def detect_vanishing_point(self, lines: np.ndarray):
        """
        Detect the vanishing point using RANSAC.

        Returns:
        --------
        numpy.ndarray
            Array of shape (3,) representing the coordinates of the detected vanishing point.
        """
        if len(lines.shape) < 2:
            # Just one line has been detected so return a point at infinity
            return [0, 1, 0]

        edges_descriptor = self.preprocess_lines(lines)
        sorted_edge_strength_indices  = np.argsort(edges_descriptor.edges_norm)
        if len(sorted_edge_strength_indices) < 2:
            return [0, 1, 0] # FIXME: this check could be redundant

        top_20_percentile_indices  = sorted_edge_strength_indices[:edges_descriptor.num_lines // 5]
        top_50_percentile_indices = sorted_edge_strength_indices[:edges_descriptor.num_lines // 2]
        if len(top_20_percentile_indices) < 1: # can happen if num_lines < 5
            if len(top_50_percentile_indices) > 1:
                top_20_percentile_indices = top_50_percentile_indices
            else:
                top_20_percentile_indices = sorted_edge_strength_indices
                top_50_percentile_indices = sorted_edge_strength_indices

        best_vanishing_point = None
        best_vote = 0
        for _ in range(self.max_iter):
            idx_1 = np.random.choice(top_20_percentile_indices )
            idx_2 = np.random.choice(top_50_percentile_indices)

            proposed_vanishing_point = np.cross(
                edges_descriptor.lines_homogeneous_eqs[idx_1],
                edges_descriptor.lines_homogeneous_eqs[idx_2]
            )
            if proposed_vanishing_point[2] == 0:
                # Vanishing point is at infinity (lines are parallel)
                continue

            proposed_vanishing_point /= proposed_vanishing_point[2]
            estimated_directions = edges_descriptor.edges_locations - proposed_vanishing_point
            dot_product = np.sum(estimated_directions * edges_descriptor.edges_directions, axis=1)
            abs_prod = np.linalg.norm(estimated_directions, axis=1)
            abs_prod[abs_prod == 0] = 1e-5

            cosine_theta = dot_product / abs_prod
            theta = np.arccos(np.abs(cosine_theta))

            vote = np.sum(
                (theta < self.theta_thresh) * edges_descriptor.edges_norm
            )

            if vote > best_vote:
                best_vote = vote
                best_vanishing_point = proposed_vanishing_point

        if best_vanishing_point is None:
            best_vanishing_point = np.array([0, 1, 0]) # Assume it's at infinity

        return best_vanishing_point
