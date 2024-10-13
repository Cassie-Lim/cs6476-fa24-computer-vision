import numpy as np
import cv2 as cv
from vision.part3_ransac import ransac_fundamental
from vision.part2_fundamental_matrix import standardize_points

def panorama_stitch(imageA, imageB, detector_mode = 'SIFT', matcher_mode = 'BF'):
    """
    ImageA and ImageB will be an image pair that you choose to stitch together
    to create your panorama. This can be your own image pair that you believe
    will give you the best stitched panorama. Feel free to play around with 
    different image pairs as a fun exercise!

    In this task, you are encouraged to explore different feature detectors 
    and matchers (e.g., SIFT, SURF, ORB, Brute-Force, FLANN) and experiment 
    to see how different techniques affect the quality of the stitched panorama.
    
    You will:
    - Detect interest points in the two images using different feature detectors.
    - Match the interest points using various feature matchers.
    - Use the matched points to compute the homography matrix.
    - Warp one of the images into the coordinate space of the other image 
      manually to create a stitched panorama (note: you may NOT use any 
      pre-existing warping function like `warpPerspective`).

    The goal is to explore how the choice of feature detectors and matchers 
    influences the final panorama quality.

    Please note that you can use your fundamental matrix estimation from part3
    (imported for you above) to compute the homography matrix that you will 
    need to stitch the panorama.
    
    Feel free to reuse your interest point pipeline from project 2, or you may
    choose to use any existing interest point/feature matching functions from
    OpenCV. You may NOT use any pre-existing warping function though.

    Args:
        imageA: first image that we are looking at (from camera view 1) [A x B]
        imageB: second image that we are looking at (from camera view 2) [M x N]

    Returns:
        panorama: stitch of image 1 and image 2 using manual warp. Ideal dimensions
            are either:
            1. A or M x (B + N)
                    OR
            2. (A + M) x B or N)
    """
    panorama = None

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    # You are encouraged to explore different feature detectors and matchers. #
    # Experiment with different techniques to find what works best for your   #
    # chosen image pair, and use them to compute the homography matrix.       #
    # Remember: You may NOT use any pre-existing warping functions!           #
    ###########################################################################
    if detector_mode == 'SIFT':
        detector = cv.SIFT_create()
    elif detector_mode == 'ORB':
        detector = cv.ORB_create()
    else:
        raise ValueError(f"Unsupported detector mode: {detector_mode}")

    keypointsA, descriptorsA = detector.detectAndCompute(imageA, None)
    keypointsB, descriptorsB = detector.detectAndCompute(imageB, None)

    if matcher_mode == 'BF':
        if detector_mode == 'SIFT':
            matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        elif detector_mode == 'ORB':
            matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    elif matcher_mode == 'FLANN':
        if detector_mode == 'SIFT':
            index_params = dict(algorithm=1, trees=5) 
            search_params = dict(checks=50)
            matcher = cv.FlannBasedMatcher(index_params, search_params)
        elif detector_mode == 'ORB':
            index_params = dict(algorithm=6, 
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
            search_params = dict(checks=50)
            matcher = cv.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError(f"Unsupported matcher mode: {matcher_mode}")

    matches = matcher.knnMatch(descriptorsA, descriptorsB, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    ptsA = np.float32([keypointsA[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    ptsB = np.float32([keypointsB[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    H, _ = cv.findHomography(ptsB, ptsA, cv.RANSAC)

    # matches_img = cv.drawMatches(imageA, keypointsA, imageB, keypointsB, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # import matplotlib.pyplot as plt
    # # Show matches
    # plt.figure(figsize=(12, 8))
    # plt.title("Feature Matches")
    # plt.imshow(matches_img)
    # plt.show()

    heightA, widthA = imageA.shape[:2]
    heightB, widthB = imageB.shape[:2]
    panorama = np.zeros((max(heightA, heightB), widthA + widthB, 3), dtype=np.uint8)
    panorama[0:heightA, 0:widthA] = imageA
    
    for y in range(heightB):
        for x in range(widthB):
            pt = np.array([x, y, 1]).reshape(3, 1)
            pt_transformed = H @ pt
            pt_transformed /= pt_transformed[2]
            x_transformed, y_transformed = int(pt_transformed[0]), int(pt_transformed[1])
            if 0 <= x_transformed < panorama.shape[1] and 0 <= y_transformed < panorama.shape[0]:
                panorama[y_transformed, x_transformed] = imageB[y, x]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return panorama

