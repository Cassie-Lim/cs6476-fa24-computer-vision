#!/usr/bin/python3

import numpy as np

def my_conv2d_freq(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply the Convolution Theorem to perform the convolution operation. 
    
    Return 
    - the input image represented in the frequency domain, 
    - the filter represented in the frequency domain,
    - the result of the convolution in the frequency domain, and 
    - the result of the convolution in the spatial domain.

    We will plot and analyze these to gain a better understanding of what is going on.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, j)
    Returns:
        image_freq: array of shape (m, n)
        filter_freq: array of shape (m, n)
        conv_result_freq: array of shape (m, n)
        conv_result: array of shape (m, n)
    HINTS:
    - Pad your filter in the spatial domain. We want to retain all of the high frequencies after the FFT
    - Return only the real component of the convolution result
    - Numpy considers frequency graphs differently than we have shown them in class. Look into the 
      documentation for np.fft.fft2 for how to account for this in the output image.
    - When applying padding, only use the zero-padding method.
    """

    ############################
        ### TODO: YOUR CODE HERE ###
    padded_filter = np.zeros_like(image)
    padded_filter[:filter.shape[0], :filter.shape[1]] = filter

    image_freq = np.fft.fft2(image)
    filter_freq = np.fft.fft2(padded_filter)
    conv_result_freq = image_freq * filter_freq 
    conv_result = np.real(np.fft.ifft2(conv_result_freq))        
    ### END OF STUDENT CODE ####
    ############################

    return image_freq, filter_freq, conv_result_freq, conv_result


def my_sharpen_freq(image: np.ndarray) -> np.ndarray:
    """
    Sharpen the input image using a sharpening filter.
    
    Return the sharpened image.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, k)
    Returns:
        sharpened_image: array of shape (m, n)
    HINTS:
    1. Apply sharpening filter to the input image using my_conv2d_freq()
        - Hint: You can use a 3x3 Laplacian filter for the sharpening
    2. Normalize the obtained sharpened image.
    3. Enchance image by adding image obtained from sharpening filter to the original image to obtain the final sharpened image
    - You should use the my_conv2d_freq function to help you with this task.
    """

    ############################
        ### TODO: YOUR CODE HERE ###
    filter = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]], dtype=np.float32) * 0.1 + \
            np.array([[0, 0, 0],
                        [0, -1, 0],
                        [0, 0, 0]], dtype=np.float32) * 4
    filter = np.array([[0,  0, -1,  0,  0],
                             [0, -1, -2, -1,  0],
                             [-1, -2, 16, -2, -1],
                             [0, -1, -2, -1,  0],
                             [0,  0, -1,  0,  0]], dtype=np.float32)
    _, _, _, aug = my_conv2d_freq(image, filter)
    sharpened_image = image + aug
    ### END OF STUDENT CODE ####
    ############################

    return sharpened_image