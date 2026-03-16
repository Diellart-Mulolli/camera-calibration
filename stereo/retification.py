
import cv2
import numpy as np

def compute_rectification_maps(calib_data):
    """
    Computes rectification transforms for both cameras.
    After rectification: epipolar lines are horizontal,
    corresponding points have same y-coordinate (yL = yR)
    """
    
    K_left = calib_data['K_left']
    D_left = calib_data['D_left']
    K_right = calib_data['K_right']
    D_right = calib_data['D_right']
    R = calib_data['R']
    T = calib_data['T']
    image_size = calib_data['image_size']
    
    # Compute rectification transforms
    # R1, R2: rectification transforms for left and right cameras
    # P1, P2: projection matrices after rectification
    # Q: disparity-to-depth mapping matrix (4x4)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left, D_left,
        K_right, D_right,
        image_size,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0  # alpha=0: no black borders, alpha=1: full image
    )
    
    # Compute remapping tables for undistortion + rectification
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left, D_left, R1, P1, image_size, cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, D_right, R2, P2, image_size, cv2.CV_32FC1
    )
    
    # Extract focal length and principal point from projection matrix P1
    # P1 = [fx  0  cx  0]
    #      [0  fy  cy  0]
    #      [0   0   1  0]
    f_rect = P1[0, 0]   # Rectified focal length (pixels)
    cx_rect = P1[0, 2]  # Rectified principal point x
    cy_rect = P1[1, 2]  # Rectified principal point y
    
    print(f"\n=== Rectification Results ===")
    print(f"Rectified focal length: f = {f_rect:.2f} px")
    print(f"Rectified principal point: ({cx_rect:.2f}, {cy_rect:.2f})")
    print(f"Q matrix (for 3D reprojection):\n{Q}")
    
    return {
        'R1': R1, 'R2': R2,
        'P1': P1, 'P2': P2,
        'Q': Q,
        'roi1': roi1, 'roi2': roi2,
        'map_left_x': map_left_x,
        'map_left_y': map_left_y,
        'map_right_x': map_right_x,
        'map_right_y': map_right_y,
        'focal_length_rect': f_rect,
        'cx_rect': cx_rect,
        'cy_rect': cy_rect
    }


def apply_rectification(img_left, img_right, rect_maps):
    """
    Apply rectification to an image pair.
    After this step: corresponding points are on same horizontal scanline
    """
    
    img_left_rect = cv2.remap(
        img_left,
        rect_maps['map_left_x'],
        rect_maps['map_left_y'],
        cv2.INTER_LANCZOS4
    )
    img_right_rect = cv2.remap(
        img_right,
        rect_maps['map_right_x'],
        rect_maps['map_right_y'],
        cv2.INTER_LANCZOS4
    )
    
    return img_left_rect, img_right_rect


def verify_rectification(img_left_rect, img_right_rect, num_lines=20):
    """
    Visual verification: draw horizontal epipolar lines.
    If rectification is correct, features should align on same line.
    """
    
    h, w = img_left_rect.shape[:2]
    
    # Create side-by-side view
    combined = np.hstack([img_left_rect, img_right_rect])
    if len(combined.shape) == 2:
        combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    
    # Draw horizontal lines to verify alignment
    step = h // num_lines
    for i in range(num_lines):
        y = i * step
        color = (0, 255, 0) if i % 2 == 0 else (0, 0, 255)
        cv2.line(combined, (0, y), (2 * w, y), color, 1)
    
    return combined