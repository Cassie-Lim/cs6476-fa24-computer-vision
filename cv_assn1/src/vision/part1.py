#!/usr/bin/python3

from typing import Tuple

import numpy as np
from vision import utils

def create_Gaussian_kernel_1D(ksize: int, sigma: int) -> np.ndarray:
    """Create a 1D Gaussian kernel using the specified filter size and standard deviation.

    The kernel should have:
    - shape (k,1)
    - mean = floor (ksize / 2)
    - values that sum to 1

    Args:
        ksize: length of kernel
        sigma: standard deviation of Gaussian distribution

    Returns:
        kernel: 1d column vector of shape (k,1)

    HINT:
    - You can evaluate the univariate Gaussian probability density function (pdf) at each
      of the 1d values on the kernel (think of a number line, with a peak at the center).
    - The goal is to discretize a 1d continuous distribution onto a vector.
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    mu = np.arange(ksize) - ksize // 2
    kernel = np.exp(-(mu**2) / (2 * sigma**2))
    kernel = kernel / (kernel.sum() + 1e-12)
    kernel = kernel.reshape(-1, 1)

    ### END OF STUDENT CODE ####
    ############################

    return kernel


def create_Gaussian_kernel_2D(cutoff_frequency: int) -> np.ndarray:
    """
    Create a 2D Gaussian kernel using the specified filter size, standard
    deviation and cutoff frequency.

    The kernel should have:
    - shape (k, k) where k = cutoff_frequency * 4 + 1
    - mean = floor(k / 2)
    - standard deviation = cutoff_frequency
    - values that sum to 1

    Args:
        cutoff_frequency: an int controlling how much low frequency to leave in
        the image.
    Returns:
        kernel: numpy nd-array of shape (k, k)

    HINT:
    - You can use create_Gaussian_kernel_1D() to complete this in one line of code.
    - The 2D Gaussian kernel here can be calculated as the outer product of two
      1D vectors. In other words, as the outer product of two vectors, each
      with values populated from evaluating the 1D Gaussian PDF at each 1d coordinate.
    - Alternatively, you can evaluate the multivariate Gaussian probability
      density function (pdf) at each of the 2d values on the kernel's grid.
    - The goal is to discretize a 2d continuous distribution onto a matrix.
    """

    ############################
    ### TODO: YOUR CODE HERE ###

    k = create_Gaussian_kernel_1D(ksize=cutoff_frequency * 4 + 1, sigma=cutoff_frequency)
    kernel = k @ k.T
    kernel = kernel / (kernel.sum() + 1e-12)

    ### END OF STUDENT CODE ####
    ############################

    return kernel


def my_conv2d_numpy(image_path: str, filter: np.ndarray) -> np.ndarray:
    """Apply a single 2d filter to each channel of an image. Return the filtered image.

    Note: we are asking you to implement a very specific type of convolution.
      The implementation in torch.nn.Conv2d is much more general.

    Args:
        image_path: string specifying the path to the input image
        filter: array of shape (k, j)
    Returns:
        filtered_image: array of shape (m, n, c)

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to
      work with matrices and functions from utils.py is fine and encouraged. Using OpenCV or similar to do
      the filtering for you is not allowed.
    - We encourage you to try implementing this naively first, just be aware
      that it may take an absurdly long time to run. You will need to get a
      function that takes a reasonable amount of time to run so that the TAs
      can verify your code works.
    - If you need to apply padding to the image, only use the zero-padding
      method. You need to compute how much padding is required, if any.
    - "Stride" should be set to 1 in your implementation.
    - You can implement either "cross-correlation" or "convolution", and the result
      will be identical, since we will only test with symmetric filters.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    ### TODO: YOUR CODE HERE ###
    image = utils.load_image(image_path)
    m, n, c = image.shape
    filtered_image = np.zeros_like(image)
    pad0 = (filter.shape[0]-1)//2
    pad1 = (filter.shape[1]-1)//2
    padded_image = np.pad(image, ((pad0, pad0), (pad1, pad1), (0, 0)), 'constant', constant_values=(0, 0))
    for i in range(m):
        for j in range(n):
            for k in range(c):
              filtered_image[i, j, k] = (
                  padded_image[i:i+filter.shape[0], j:j+filter.shape[1], k] * filter
              ).sum()
    ### END OF STUDENT CODE ####
    ############################

    return filtered_image


def create_hybrid_image(
    image_path1: str, image_path2: str, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.

    Args:
        image_path1: string specifying the path to the input image
        image_path2: string specifying the path to the input image
        filter: array of dim (x, y)
    Returns:
        low_frequencies: array of shape (m, n, c)
        high_frequencies: array of shape (m, n, c)
        hybrid_image: array of shape (m, n, c)

    HINTS:
    - You will use your my_conv2d_numpy() function in this function.
    - You can get just the high frequency content of an image by removing its
      low frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values of the hybrid image are
      between 0 and 1. This is known as 'clipping'.
    - If you want to use images with different dimensions, you should resize
      them in the notebook code.
    """

    ############################
    ### TODO: YOUR CODE HERE ###

    low_frequencies = my_conv2d_numpy(image_path1, filter)

    low_freq2 = my_conv2d_numpy(image_path2, filter)
    img2 = utils.load_image(image_path2)
    high_frequencies = img2 - low_freq2

    hybrid_image = np.clip(low_frequencies + high_frequencies, 0, 1)

    ### END OF STUDENT CODE ####
    ############################

    return low_frequencies, high_frequencies, hybrid_image
