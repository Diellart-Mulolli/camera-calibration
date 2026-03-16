# camera_calibration.py
# Single-camera calibration using Zhang's method
# Paper Section 4.3

import cv2
import numpy as np
import glob
import os


def calibrate_single_camera(
    image_folder: str,
    board_size: tuple = (9, 6),
    square_size: float = 0.025,
    save_path: str = None,
    verbose: bool = True
) -> dict:
    """
    Zhang's single camera calibration.

    Args:
        image_folder : path to folder with chessboard images
        board_size   : (cols, rows) inner corners
        square_size  : physical square size in meters
        save_path    : if given, saves .npz calibration file
        verbose      : print progress

    Returns:
        dict with K, D, rvecs, tvecs, rms, image_size
    """

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare 3D object points
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0],
                            0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []   # 3D world points
    imgpoints = []   # 2D image points

    images = sorted(
        glob.glob(os.path.join(image_folder, '*.png')) +
        glob.glob(os.path.join(image_folder, '*.jpg'))
    )

    if len(images) == 0:
        raise FileNotFoundError(f"No images found in: {image_folder}")

    img_shape = None
    valid_count = 0

    for fname in images:
        img  = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            objpoints.append(objp)
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            imgpoints.append(corners_refined)
            valid_count += 1
            if verbose:
                print(f"  ✓ {os.path.basename(fname)}")
        else:
            if verbose:
                print(f"  ✗ {os.path.basename(fname)} — corners not found")

    if valid_count < 5:
        raise ValueError(f"Need at least 5 valid images, got {valid_count}")

    # Calibrate
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    result = {
        'K': K,
        'D': D,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'rms': rms,
        'image_size': img_shape,
        'valid_images': valid_count,
        'total_images': len(images)
    }

    if verbose:
        print(f"\n  ── Calibration Results ──")
        print(f"  Valid images : {valid_count}/{len(images)}")
        print(f"  RMS error    : {rms:.4f} px")
        print(f"  Focal (fx)   : {K[0,0]:.2f} px")
        print(f"  Focal (fy)   : {K[1,1]:.2f} px")
        print(f"  Principal (cx, cy): ({K[0,2]:.1f}, {K[1,2]:.1f})")
        print(f"  Distortion k1: {D[0,0]:.6f}")

    if save_path:
        np.savez(save_path, **result)
        print(f"  Saved → {save_path}")

    return result


def compute_reprojection_error(result: dict) -> float:
    """Compute mean reprojection error from calibration result."""
    total_error = 0
    for i in range(len(result['rvecs'])):
        imgpoints2, _ = cv2.projectPoints(
            result['objpoints'][i] if 'objpoints' in result else None,
            result['rvecs'][i],
            result['tvecs'][i],
            result['K'],
            result['D']
        )
    return total_error


if __name__ == "__main__":
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "calibration_images/left"
    print(f"Calibrating from: {folder}")

    result = calibrate_single_camera(
        image_folder=folder,
        board_size=(9, 6),
        square_size=0.025,
        save_path="output/camera_calibration.npz",
        verbose=True
    )
    print("\nDone.")