# stereo_calibration.py
# Stereo camera calibration — Paper Section 4.3 & 4.4
# Student: Mërgim Pirraku | UBT 2026

import cv2
import numpy as np
import glob
import os


def run_stereo_calibration(
    left_folder: str,
    right_folder: str,
    board_size: tuple = (9, 6),
    square_size: float = 0.025,
    save_path: str = "output/stereo_calibration.npz",
    verbose: bool = True
) -> dict | None:
    """
    Full stereo calibration pipeline.

    Steps:
        1. Detect chessboard corners in all image pairs
        2. Run individual camera calibration (Zhang's method)
        3. Run stereoCalibrate with CALIB_FIX_INTRINSIC
        4. Compute stereoRectify maps
        5. Save everything to .npz

    Returns:
        calibration dict or None on failure
    """

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3D object points
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0],
                            0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints  = []
    imgpts_l   = []
    imgpts_r   = []

    left_imgs  = sorted(
        glob.glob(os.path.join(left_folder,  '*.png')) +
        glob.glob(os.path.join(left_folder,  '*.jpg'))
    )
    right_imgs = sorted(
        glob.glob(os.path.join(right_folder, '*.png')) +
        glob.glob(os.path.join(right_folder, '*.jpg'))
    )

    if len(left_imgs) == 0:
        print(f"ERROR: No left images in {left_folder}")
        return None

    if len(left_imgs) != len(right_imgs):
        print(f"ERROR: Image count mismatch "
              f"(L={len(left_imgs)}, R={len(right_imgs)})")
        return None

    img_shape = None
    valid = 0

    print(f"Processing {len(left_imgs)} image pairs...")

    for lp, rp in zip(left_imgs, right_imgs):
        gl = cv2.cvtColor(cv2.imread(lp), cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2GRAY)
        img_shape = gl.shape[::-1]

        rl, cl = cv2.findChessboardCorners(gl, board_size, None)
        rr, cr = cv2.findChessboardCorners(gr, board_size, None)

        if rl and rr:
            objpoints.append(objp)
            imgpts_l.append(cv2.cornerSubPix(gl, cl, (11,11), (-1,-1), criteria))
            imgpts_r.append(cv2.cornerSubPix(gr, cr, (11,11), (-1,-1), criteria))
            valid += 1
            if verbose:
                print(f"  ✓ Pair {valid:02d}: {os.path.basename(lp)}")
        else:
            if verbose:
                print(f"  ✗ Skip: {os.path.basename(lp)}")

    if valid < 5:
        print(f"ERROR: Only {valid} valid pairs. Need at least 5.")
        return None

    print(f"\nValid pairs: {valid}/{len(left_imgs)}")

    # ── Individual calibrations ──────────────────────────────
    print("\nCalibrating left camera...")
    _, Kl, Dl, _, _ = cv2.calibrateCamera(
        objpoints, imgpts_l, img_shape, None, None
    )

    print("Calibrating right camera...")
    _, Kr, Dr, _, _ = cv2.calibrateCamera(
        objpoints, imgpts_r, img_shape, None, None
    )

    # ── Stereo calibration ───────────────────────────────────
    print("\nRunning stereoCalibrate...")
    rms, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpts_l, imgpts_r,
        Kl, Dl, Kr, Dr, img_shape,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=criteria
    )

    baseline = float(np.linalg.norm(T))

    # ── Rectification ────────────────────────────────────────
    print("Computing rectification maps...")
    R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
        Kl, Dl, Kr, Dr, img_shape, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    map_lx, map_ly = cv2.initUndistortRectifyMap(
        Kl, Dl, R1, P1, img_shape, cv2.CV_32FC1
    )
    map_rx, map_ry = cv2.initUndistortRectifyMap(
        Kr, Dr, R2, P2, img_shape, cv2.CV_32FC1
    )

    result = {
        # Intrinsics
        'K_left':  Kl, 'D_left':  Dl,
        'K_right': Kr, 'D_right': Dr,
        # Extrinsics
        'R': R, 'T': T, 'E': E, 'F': F,
        # Rectification
        'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
        'map_left_x':  map_lx, 'map_left_y':  map_ly,
        'map_right_x': map_rx, 'map_right_y': map_ry,
        # Metrics
        'baseline':   np.array([baseline]),
        'image_size': np.array(img_shape),
        'rms':        np.array([rms]),
        'focal_rect': np.array([float(P1[0, 0])])
    }

    # ── Print summary ────────────────────────────────────────
    if verbose:
        print(f"\n{'─'*45}")
        print(f"  Stereo Calibration Results")
        print(f"{'─'*45}")
        print(f"  RMS reprojection error : {rms:.4f} px")
        print(f"  Focal length (left fx) : {Kl[0,0]:.2f} px")
        print(f"  Focal rect  (P1[0,0])  : {P1[0,0]:.2f} px")
        print(f"  Baseline               : {baseline*100:.3f} cm")
        print(f"  Translation T          : {T.T}")
        print(f"  Image size             : {img_shape}")
        print(f"{'─'*45}")

    # ── Save ─────────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **result)
        print(f"\n  Saved → {save_path}")

    return result


if __name__ == "__main__":
    result = run_stereo_calibration(
        left_folder="calibration_images/left",
        right_folder="calibration_images/right",
        board_size=(9, 6),
        square_size=0.025,
        save_path="output/stereo_calibration.npz"
    )