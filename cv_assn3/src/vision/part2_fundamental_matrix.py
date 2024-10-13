"""Fundamental matrix utilities."""

import numpy as np

def compute_fundamental_matrix(
    pts_a: np.ndarray, pts_b: np.ndarray
) -> np.ndarray:
    """
    Estimates the fundamental matrix using point correspondences between two images.

    Args:
        pts_a: A numpy array of shape (N, 2) representing points in image A
        pts_b: A numpy array of shape (N, 2) representing points in image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################
    pts_a_normalized, T_a = standardize_points(pts_a)
    pts_b_normalized, T_b = standardize_points(pts_b)
    
    A = []
    for pa, pb in zip(pts_a_normalized, pts_b_normalized):
        x_a, y_a = pa
        x_b, y_b = pb
        A.append([x_b * x_a, x_b * y_a, x_b, y_b * x_a, y_b * y_a, y_b, x_a, y_a, 1])
    
    A = np.array(A)


    # # debug
    # _, s_vals, _ = np.linalg.svd(A)
    # condition_number = s_vals[0] / s_vals[-1]
    # print("Condition number of A:", condition_number, s_vals.shape)

    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0 
    F = U @ np.diag(S) @ Vt
    
    F = denormalize_fundamental_matrix(F, T_a, T_b)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F

def denormalize_fundamental_matrix(
    F_normalized: np.ndarray, T_a: np.ndarray, T_b: np.ndarray
) -> np.ndarray:
    """
    Adjusts the normalized fundamental matrix using the transformation matrices.

    Args:
        F_normalized: A numpy array of shape (3, 3) representing the normalized fundamental matrix
        T_a: Transformation matrix for image A
        T_b: Transformation matrix for image B

    Returns:
        F_denormalized: A numpy array of shape (3, 3) representing the original fundamental matrix
    """
    ###########################################################################
    F_denormalized = T_b.T @ F_normalized @ T_a                                 #
    ###########################################################################

    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_denormalized

def standardize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Normalizes 2D points to improve numerical stability in computations.

    Args:
        points: A numpy array of shape (N, 2) representing the 2D points

    Returns:
        points_standardized: A numpy array of shape (N, 2) representing the normalized 2D points
        T: The transformation matrix used for normalization
    """
    ###########################################################################
    ###########################################################################
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    T = np.array([[1/std[0], 0, -mean[0]/std[0]],
                  [0, 1/std[1], -mean[1]/std[1]],
                  [0, 0, 1]])
    
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_standardized = (T @ points_homogeneous.T).T[:, :2]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_standardized, T
