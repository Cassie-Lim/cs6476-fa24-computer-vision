import math
import numpy as np
from vision.part2_fundamental_matrix import compute_fundamental_matrix

def compute_ransac_iterations(
    success_prob: float, sample_count: int, inlier_prob: float
) -> int:
    """
    Computes the number of RANSAC iterations needed to achieve a certain success probability.

    Args:
        success_prob: Desired probability of successful estimation
        sample_count: Number of points sampled in each iteration
        inlier_prob: Probability that a single point is an inlier

    Returns:
        num_iterations: Number of RANSAC iterations required
    """
    num_iterations = None
    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################
    if inlier_prob == 1.0:
        num_iterations = 1
    else:
        num_iterations = np.log(1 - success_prob) / np.log(1 - inlier_prob ** sample_count)



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_iterations)

def ransac_fundamental(
    pts_a: np.ndarray, pts_b: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Uses RANSAC to estimate the fundamental matrix between two sets of points.

    Tips:
        - Determine appropriate values for success_prob, sample_count, and inlier_prob.
        - Use numpy.random.choice to select random samples.
        - An error threshold of 0.1 is recommended for distinguishing inliers.

    Args:
        pts_a: An array of shape (N, 2) containing points from image A
        pts_b: An array of shape (N, 2) containing points from image B

    Returns:
        best_fundamental_matrix: The estimated fundamental matrix
        inlier_pts_a: Inlier points from image A
        inlier_pts_b: Inlier points from image B
    """
    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################
    success_prob = 0.995
    sample_count = 8
    inlier_prob = 0.5
    error_threshold = 0.1
    num_points = pts_a.shape[0]
    num_iterations = compute_ransac_iterations(success_prob, sample_count, inlier_prob)
    best_fundamental_matrix = None
    best_inlier_count = 0
    inlier_pts_a = None
    inlier_pts_b = None
    print("Num iter for ransac", num_iterations)
    for _ in range(num_iterations):
        sample_indices = np.random.choice(num_points, sample_count, replace=False)
        sample_pts_a = pts_a[sample_indices]
        sample_pts_b = pts_b[sample_indices]
        F = compute_fundamental_matrix(sample_pts_a, sample_pts_b)

        pts_a_homogeneous = np.hstack((pts_a, np.ones((pts_a.shape[0], 1))))
        pts_b_homogeneous = np.hstack((pts_b, np.ones((pts_b.shape[0], 1))))
        lines_a = F.T @ pts_b_homogeneous.T
        errors = np.abs(np.sum(pts_a_homogeneous * lines_a.T, axis=1)) / np.sqrt(lines_a[0]**2 + lines_a[1]**2)
    
        inlier_indices = np.where(errors < error_threshold)[0]
        inlier_count = len(inlier_indices)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_fundamental_matrix = F
            inlier_pts_a = pts_a[inlier_indices]
            inlier_pts_b = pts_b[inlier_indices]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_fundamental_matrix, inlier_pts_a, inlier_pts_b
