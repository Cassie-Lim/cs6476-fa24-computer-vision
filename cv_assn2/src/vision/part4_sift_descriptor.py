#!/usr/bin/python3

import copy
import pdb
import time
from typing import Tuple
import numpy as np

import torch

from src.vision.part1_harris_corner import compute_image_gradients, SCHARR_X_KERNEL, SCHARR_Y_KERNEL
from torch import nn


"""
Implement SIFT  (See Szeliski 7.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

Your implementation will not exactly match the SIFT reference. For example,
we will be excluding scale and rotation invariance.

You do not need to perform the interpolation in which each gradient
measurement contributes to multiple orientation bins in multiple cells.
"""


def get_orientations_and_magnitudes(
    Ix: np.ndarray,
    Iy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function will return the orientations and magnitudes of the
    gradients at each pixel location.

    Args:
        Ix: array of shape (m,n), representing x gradients in the image
        Iy: array of shape (m,n), representing y gradients in the image
    Returns:
        orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from
            -PI to PI.
        magnitudes: A numpy array of shape (m,n), representing magnitudes of
            the gradients at each pixel location
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    magnitudes = np.sqrt(Ix**2 + Iy**2)
    orientations = np.arctan2(Iy, Ix)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return orientations, magnitudes


def get_gradient_histogram_vec_from_patch(
    window_orientations: np.ndarray,
    window_magnitudes: np.ndarray
) -> np.ndarray:
    """ Given 16x16 patch, form a 128-d vector of gradient histograms.

    Key properties to implement:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the terminology
        used in the feature literature to describe the spatial bins where
        gradient distributions will be described. The grid will extend
        feature_width/2 - 1 to the left of the "center", and feature_width/2 to
        the right. The same applies to above and below, respectively.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be
        added to the feature vector left to right then row by row (reading
        order).

    Do not normalize the histogram here to unit norm -- preserve the histogram
    values. A useful function to look at would be np.histogram.

    Args:
        window_orientations: (16,16) array representing gradient orientations of
            the patch
        window_magnitudes: (16,16) array representing gradient magnitudes of the
            patch

    Returns:
        wgh: (128,1) representing weighted gradient histograms for all 16
            neighborhoods of size 4x4 px
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    wgh = []
    # bin_centers = np.linspace(-7 * np.pi / 8, 7 * np.pi / 8, 8)  # 8 centers
    # bin_edges = np.concatenate(([-np.pi], (bin_centers[:-1] + bin_centers[1:]) / 2, [np.pi]))
    bin_edges = np.linspace(-np.pi, np.pi, 9) - 1e-6
    for i in range(4):
        for j in range(4):
            local_window_orientations = window_orientations[i*4:(i+1)*4, j*4:(j+1)*4].flatten()
            local_window_magnitudes = window_magnitudes[i*4:(i+1)*4, j*4:(j+1)*4].flatten()
            hist, _ = np.histogram(local_window_orientations, bins=bin_edges, weights=local_window_magnitudes)
            # print(hist, len(local_window_orientations), local_window_magnitudes)
            wgh.append(hist)
    wgh = np.array(wgh).reshape((-1, 1))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return wgh


def get_feat_vec(
    x: float,
    y: float,
    orientations,
    magnitudes,
    feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the feature vector for a specific interest point.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)
    Your implementation does not need to exactly match the SIFT reference.


    Your (baseline) descriptor should have:
    (1) Each feature should be normalized to unit length.
    (2) Each feature should be raised to the 1/2 power, i.e. square-root SIFT
        (read https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)

    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions.
    The autograder will only check for each gradient contributing to a single bin.

    Args:
        x: a float, the column (x-coordinate) of the interest point
        y: A float, the row (y-coordinate) of the interest point
        orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
        magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    Returns:
        fv: A numpy array of shape (feat_dim,1) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    delta = feature_width//2 - (feature_width+1)%2
    start_y = y - delta
    start_x = x -delta
    if start_y<0 or start_x<0 or start_y+feature_width>orientations.shape[0] or start_x+feature_width>orientations.shape[1]:
        print("warning: the index exceed the boundary in get_feat_vec")
    patch_orientations = orientations[start_y:start_y+feature_width, start_x:start_x+feature_width]
    patch_magnitudes = magnitudes[start_y:start_y+feature_width, start_x:start_x+feature_width]
    fv = get_gradient_histogram_vec_from_patch(patch_orientations, patch_magnitudes)
    """
    ROOTSIFT from original paper:  
    (i)L1 normalize the SIFT vector (originally it has unit L2 norm); 
    (ii) square root each element. It then follows that
    SE(√x,√y) = √xT√y =H(x,y),and the resulting vectors are L2 normalized since SE(√x,√x) = Pn
    i xi = 1.
    We thus define a new descriptor, which we term RootSIFT,
    which is an element wise square root of the L1 normalized
    SIFT vectors.
    """
    fv = fv / (np.linalg.norm(fv, ord=2)+1e-12)  # TODO: it seems the test will fail with l1 norm
    fv = np.sqrt(fv)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv


def get_SIFT_descriptors(
    image_bw: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the 128-d SIFT features computed at each of the input
    points. Implement the more effective SIFT descriptor (see Szeliski 7.1.2 or
    the original publications at http://www.cs.ubc.ca/~lowe/keypoints/)

    Args:
        image: A numpy array of shape (m,n), the image
        X: A numpy array of shape (k,), the x-coordinates of interest points
        Y: A numpy array of shape (k,), the y-coordinates of interest points
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e.,
            every cell of your local SIFT-like feature will have an integer
            width and height). This is the initial window size we examine
            around each keypoint.
    Returns:
        fvs: A numpy array of shape (k, feat_dim) representing all feature
            vectors. "feat_dim" is the feature_dimensionality (e.g., 128 for
            standard SIFT). These are the computed features.
    """
    assert image_bw.ndim == 2, 'Image must be grayscale'

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    Ix, Iy = compute_image_gradients(image_bw)
    orientations, magnitudes = get_orientations_and_magnitudes(Ix, Iy)
    fvs = []
    for x, y in zip(X, Y):
        fv = get_feat_vec(x, y, orientations, magnitudes, feature_width)
        fvs.append(fv.flatten())
    fvs = np.array(fvs)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return fvs


### ----------------- OPTIONAL (below) ------------------------------------

## Implementation of the function below is  optional (extra credit)

def get_sift_features_vectorized(
    image_bw: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    This function is a vectorized version of `get_SIFT_descriptors`.

    As before, start by computing the image gradients, as done before. Then
    using PyTorch convolution with the appropriate weights, create an output
    with 10 channels, where the first 8 represent cosine values of angles
    between unit circle basis vectors and image gradient vectors at every
    pixel. The last two channels will represent the (dx, dy) coordinates of the
    image gradient at this pixel. The gradient at each pixel can be projected
    onto 8 basis vectors around the unit circle

    Next, the weighted histogram can be created by element-wise multiplication
    of a 4d gradient magnitude tensor, and a 4d gradient binary occupancy
    tensor, where a tensor cell is activated if its value represents the
    maximum channel value within a "fibre" (see
    http://cs231n.github.io/convolutional-networks/ for an explanation of a
    "fibre"). There will be a fibre (consisting of all channels) at each of the
    (M,N) pixels of the "feature map".

    The four dimensions represent (N,C,H,W) for batch dim, channel dim, height
    dim, and weight dim, respectively. Our batch size will be 1.

    In order to create the 4d binary occupancy tensor, you may wish to index in
    at many values simultaneously in the 4d tensor, and read or write to each
    of them simultaneously. This can be done by passing a 1D PyTorch Tensor for
    every dimension, e.g., by following the syntax:
        My4dTensor[dim0_idxs, dim1_idxs, dim2_idxs, dim3_idxs] = 1d_tensor.

    Finally, given 8d feature vectors at each pixel, the features should be
    accumulated over 4x4 subgrids using PyTorch convolution.

    You may find torch.argmax(), torch.zeros_like(), torch.meshgrid(),
    flatten(), torch.arange(), torch.unsqueeze(), torch.mul(), and
    torch.norm() helpful.

    Returns:
        fvs
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    H, W = image_bw.shape

    image_tensor = torch.from_numpy(image_bw).unsqueeze(0).unsqueeze(0).float()

    scharr_x = torch.from_numpy(SCHARR_X_KERNEL).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)
    scharr_y = torch.from_numpy(SCHARR_Y_KERNEL).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)

    Ix = nn.functional.conv2d(image_tensor, scharr_x, padding=1).squeeze()
    Iy = nn.functional.conv2d(image_tensor, scharr_y, padding=1).squeeze()


    magnitudes = torch.sqrt(Ix**2 + Iy**2)
    orientations = torch.arctan2(Iy, Ix)
    fvs = []

    angles = torch.linspace(-np.pi*7/8, np.pi*7/8, 8) 
    cos_projections = torch.cos(angles.unsqueeze(-1) - orientations.reshape(-1)).reshape(8, H, W)
    print(cos_projections.shape, angles/np.pi)

    feature_map = torch.zeros((1, 10, H, W), dtype=torch.float32)
    feature_map[:, :8, :, :] = cos_projections
    feature_map[:, 8, :, :] = Ix
    feature_map[:, 9, :, :] = Iy

    occupancy_tensor = (cos_projections == torch.max(cos_projections, dim=0, keepdim=True)[0])

    weighted_histogram = (magnitudes * occupancy_tensor).unsqueeze(0)
    print(weighted_histogram.shape, occupancy_tensor.shape, magnitudes.shape, occupancy_tensor.sum(0).float().mean())

    fvs = []
    feature_width = 16
    delta = feature_width//2 - (feature_width+1)%2
    grid_size = 4
    weight = torch.ones(8, 1, grid_size, grid_size, dtype=torch.float32)
    for x, y in zip(X, Y):
        start_y = y - delta
        start_x = x -delta
        if start_y < 0 or start_x < 0 or start_y + feature_width > H or start_x + feature_width > W:
            continue 
        patch = weighted_histogram[:, :, start_y:start_y+feature_width, start_x:start_x+feature_width]
        fv = nn.functional.conv2d(patch, weight, stride=grid_size, groups=8).flatten()
        fv = fv / (torch.norm(fv)+1e-6)  
        fv = torch.sqrt(fv)
        fvs.append(fv)
    fvs = torch.stack(fvs).squeeze().numpy()
    
    print("SIFT features shape", fvs.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
