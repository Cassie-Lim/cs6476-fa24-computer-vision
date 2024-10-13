import numpy as np

def compute_camera_center(P: np.ndarray) -> np.ndarray:
    """
    Computes the camera center location from a given projection matrix.

    Args:
        P: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
        camera_center: A numpy array of shape (1, 3) representing the camera center
                       location in world coordinates
    """
    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################
    M = P[:, :3]
    p_4 = P[:, 3]
    camera_center = -np.linalg.inv(M) @ p_4
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return camera_center

def project_points(M: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Projects 3D points into 2D image coordinates using the projection matrix.

    Args:
        M: A 3 x 4 numpy array representing the projection matrix
        points_3d: A numpy array of shape (N, 3) representing 3D points

    Returns:
        points_2d: A numpy array of shape (N, 2) representing projected 2D points
    """
    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_2d_homogeneous = M @ points_3d_homogeneous.T
    points_2d = (points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]).T


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_2d

def compute_projection_matrix(image_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    """
    Computes the projection matrix from corresponding 2D-3D point pairs.

    To compute the projection matrix, set up a system of equations using the known 2D and 3D point correspondences. You can then solve for the projection matrix using least squares or SVD. Note that each point pair provides two equations, and at least 6 point pairs are needed to solve for the projection matrix.

    Args:
        image_points: A numpy array of shape (N, 2)
        world_points: A numpy array of shape (N, 3)

    Returns:
        P: A numpy array of shape (3, 4) representing the projection matrix
    """
    ###########################################################################
    # TODO: Implement this function                                           #
    ###########################################################################
    num_points = image_points.shape[0]
    if num_points < 6:
        raise ValueError("At least 6 point pairs are required to compute the projection matrix")

    A = []

    for i in range(num_points):
        X, Y, Z = world_points[i]
        x, y = image_points[i]
        A.append([-X, -Y, -Z, -1,  0,  0,  0,  0,  x*X,  x*Y,  x*Z,  x])
        A.append([ 0,  0,  0,  0, -X, -Y, -Z, -1,  y*X,  y*Y,  y*Z,  y])

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1, :].reshape(3, 4)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return P
