# utils/image_utils.py
# Shared image utilities for stereo pipeline

import cv2
import numpy as np
import os


def load_stereo_pair(folder: str):
    """
    Load a stereo pair from a folder.
    Tries: left.png/right.png, im0.png/im1.png
    """
    candidates = [
        ('left.png',  'right.png'),
        ('left.jpg',  'right.jpg'),
        ('im0.png',   'im1.png'),
        ('im0.jpg',   'im1.jpg'),
    ]

    for l_name, r_name in candidates:
        lp = os.path.join(folder, l_name)
        rp = os.path.join(folder, r_name)
        if os.path.exists(lp) and os.path.exists(rp):
            return cv2.imread(lp), cv2.imread(rp)

    raise FileNotFoundError(f"No stereo pair found in: {folder}")


def to_colormap(img_np: np.ndarray, colormap=cv2.COLORMAP_PLASMA,
                clip_percentile: float = 95) -> np.ndarray:
    """Apply colormap to float/nan image."""
    out = np.nan_to_num(img_np, nan=0).astype(np.float32)
    if clip_percentile < 100:
        valid = out[out > 0]
        if len(valid):
            p = np.percentile(valid, clip_percentile)
            out = np.clip(out, 0, p)
    norm = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(norm, colormap)


def resize_keep_aspect(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Resize image preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def draw_epipolar_lines(left: np.ndarray, right: np.ndarray,
                         n_lines: int = 15) -> np.ndarray:
    """Draw horizontal epipolar lines on rectified pair."""
    combined = np.hstack([left.copy(), right.copy()])
    if len(combined.shape) == 2:
        combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    h = combined.shape[0]
    for i in range(0, n_lines):
        y = int(i * h / n_lines)
        cv2.line(combined, (0, y), (combined.shape[1], y),
                 (0, 200, 80), 1)
    return combined