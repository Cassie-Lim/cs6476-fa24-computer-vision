#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    fvs = []
    delta_half = feature_width//2 - (feature_width+1)%2
    
    for x, y in zip(X, Y):
        y_start = y - delta_half
        x_start = x - delta_half
        local_image_vec = image_bw[y_start:y_start+feature_width, x_start:x_start+feature_width].flatten()
        local_image_vec = local_image_vec / (np.linalg.norm(local_image_vec)+1e-6)
        fvs.append(local_image_vec)
    fvs = np.array(fvs)
    print(X.shape, fvs.shape)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
