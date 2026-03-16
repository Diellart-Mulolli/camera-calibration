# d = xL - xR = fB/Z

import cv2
import numpy as np

def compute_disparity_sgbm(img_left_gray, img_right_gray,
                            min_disparity=0,
                            num_disparities=128,
                            block_size=11):
    """
    Semi-Global Block Matching (SGBM) - higher quality than basic BM
    Implements: d = xL - xR  (Section 5.7)
    """
    
    # SGBM Parameters
    window_size = block_size
    
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,     # Must be divisible by 16
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,        # Smoothness penalty (small changes)
        P2=32 * 3 * window_size ** 2,       # Smoothness penalty (large changes)
        disp12MaxDiff=1,                    # Max allowed diff in left-right check
        uniquenessRatio=15,                 # Uniqueness of best match
        speckleWindowSize=100,              # Noise filter window
        speckleRange=32,                    # Noise filter range
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Left disparity map
    disparity_left = left_matcher.compute(img_left_gray, img_right_gray)
    
    # Right disparity map for Left-Right consistency check
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    disparity_right = right_matcher.compute(img_right_gray, img_left_gray)
    
    # WLS Filter for smoother results
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)
    
    disparity_filtered = wls_filter.filter(
        disparity_left, img_left_gray,
        disparity_map_right=disparity_right
    )
    
    # Convert from fixed-point (multiply by 16 internally) to float
    disparity_float = disparity_filtered.astype(np.float32) / 16.0
    
    return disparity_float


def compute_disparity_bm(img_left_gray, img_right_gray,
                          num_disparities=64,
                          block_size=15):
    """
    Basic Block Matching - simpler, faster (used in your paper's implementation)
    Direct equivalent of the code shown in Section 7
    """
    
    stereo = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size
    )
    
    disparity = stereo.compute(img_left_gray, img_right_gray)
    disparity_float = disparity.astype(np.float32) / 16.0
    
    return disparity_float


def filter_disparity(disparity, min_disp=1.0, max_disp=None):
    """
    Filter invalid disparity values.
    d=0 or d<0 means no valid match was found.
    """
    filtered = disparity.copy()
    
    # Mask invalid disparities
    invalid_mask = disparity <= min_disp
    if max_disp is not None:
        invalid_mask |= disparity > max_disp
    
    filtered[invalid_mask] = np.nan  # Mark as invalid
    
    valid_count = np.sum(~invalid_mask)
    total_count = disparity.size
    print(f"Valid disparity pixels: {valid_count}/{total_count} "
          f"({100*valid_count/total_count:.1f}%)")
    
    return filtered, invalid_mask