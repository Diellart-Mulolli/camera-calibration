# datasets/download_datasets.py
# Auto-downloads Middlebury stereo datasets for the project
# Student: Mërgim Pirraku | UBT 2026

import os
import sys
import zipfile
import tarfile
import urllib.request
import urllib.error
import shutil
import json
from pathlib import Path

# ============================================================
# DATASET REGISTRY
# ============================================================

DATASETS = {
    # ── Middlebury 2001 (Classic, small, fast) ───────────────
    "tsukuba": {
        "name": "Tsukuba",
        "description": "Indoor scene, near range ~0.5-1.5m",
        "url": "https://vision.middlebury.edu/stereo/data/scenes2001/data/tsukuba.zip",
        "type": "middlebury_2001",
        "ground_truth": True,
        "image_size": "384x288",
        "disparity_range": "0-16",
        "paper_section": "§5.7, §5.8",
    },
    "venus": {
        "name": "Venus",
        "description": "Smooth disparity gradient, good for error analysis",
        "url": "https://vision.middlebury.edu/stereo/data/scenes2001/data/venus.zip",
        "type": "middlebury_2001",
        "ground_truth": True,
        "image_size": "434x383",
        "disparity_range": "0-20",
        "paper_section": "§6",
    },
    "map": {
        "name": "Map",
        "description": "Fronto-parallel plane, calibration testing",
        "url": "https://vision.middlebury.edu/stereo/data/scenes2001/data/map.zip",
        "type": "middlebury_2001",
        "ground_truth": True,
        "image_size": "284x216",
        "disparity_range": "0-30",
        "paper_section": "§4.3",
    },

    # ── Middlebury 2003 ──────────────────────────────────────
    "cones": {
        "name": "Cones",
        "description": "Classic benchmark, varied depth",
        "url": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/cones.zip",
        "type": "middlebury_2003",
        "ground_truth": True,
        "image_size": "450x375",
        "disparity_range": "0-255",
        "paper_section": "§5.7, §5.8",
    },
    "teddy": {
        "name": "Teddy",
        "description": "Most used benchmark, good depth variation",
        "url": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/teddy/teddy.zip",
        "type": "middlebury_2003",
        "ground_truth": True,
        "image_size": "450x375",
        "disparity_range": "0-255",
        "paper_section": "§5.7, §5.8, §6",
    },

    # ── Middlebury 2005 ──────────────────────────────────────
    "art": {
        "name": "Art",
        "description": "High resolution, complex scene",
        "url": "https://vision.middlebury.edu/stereo/data/scenes2005/HighRes/Art.zip",
        "type": "middlebury_2005",
        "ground_truth": True,
        "image_size": "1390x1110",
        "disparity_range": "0-512",
        "paper_section": "§5",
    },
    "books": {
        "name": "Books",
        "description": "Books scene, good for disparity analysis",
        "url": "https://vision.middlebury.edu/stereo/data/scenes2005/HighRes/Books.zip",
        "type": "middlebury_2005",
        "ground_truth": True,
        "image_size": "1390x1110",
        "disparity_range": "0-256",
        "paper_section": "§5",
    },

    # ── Calibration patterns (synthetic) ────────────────────
    "calib_synthetic": {
        "name": "Synthetic Calibration Images",
        "description": "Pre-generated chessboard patterns for calibration",
        "url": None,  # Generated locally
        "type": "synthetic_calib",
        "ground_truth": False,
        "image_size": "640x480",
        "disparity_range": "N/A",
        "paper_section": "§4.3",
    },
}

# ============================================================
# DOWNLOADER
# ============================================================

class DatasetDownloader:

    def __init__(self, base_dir=None):
        if base_dir is None:
            # Auto-detect project root
            script_dir = Path(__file__).parent
            base_dir   = script_dir.parent / "datasets"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    def download_all(self, keys=None, log_fn=print):
        keys = keys or list(DATASETS.keys())
        results = {}
        for key in keys:
            if key == "calib_synthetic":
                ok = self.generate_calibration_images(log_fn=log_fn)
            else:
                ok = self.download_dataset(key, log_fn=log_fn)
            results[key] = ok
        return results

    # ----------------------------------------------------------
    def download_dataset(self, key, log_fn=print):
        if key not in DATASETS:
            log_fn(f"ERROR: Unknown dataset '{key}'")
            return False

        info   = DATASETS[key]
        outdir = self.base_dir / key

        if self._is_complete(outdir, info):
            log_fn(f"  ✓ {info['name']} already downloaded, skipping.")
            return True

        outdir.mkdir(parents=True, exist_ok=True)

        if info["url"] is None:
            return False

        zip_path = outdir / f"{key}.zip"
        log_fn(f"\n  ⬇  Downloading {info['name']}...")
        log_fn(f"     {info['url']}")

        try:
            self._download_with_progress(info["url"], zip_path, log_fn)
            log_fn(f"  📦 Extracting {info['name']}...")
            self._extract(zip_path, outdir)
            zip_path.unlink(missing_ok=True)
            self._normalize_structure(outdir, info, log_fn)
            self._save_meta(outdir, info)
            log_fn(f"  ✓ {info['name']} ready at: {outdir}")
            return True

        except Exception as e:
            log_fn(f"  ✗ Failed: {e}")
            return False

    # ----------------------------------------------------------
    def _download_with_progress(self, url, dest, log_fn):
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = min(count * block_size * 100 / total_size, 100)
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                log_fn(f"\r     [{bar}] {pct:.1f}%", end="")

        urllib.request.urlretrieve(url, dest, reporthook)
        log_fn("")  # newline after progress

    # ----------------------------------------------------------
    def _extract(self, zip_path, outdir):
        if zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(outdir)
        elif tarfile.is_tarfile(str(zip_path)):
            with tarfile.open(zip_path, 'r:*') as t:
                t.extractall(outdir)

    # ----------------------------------------------------------
    def _normalize_structure(self, outdir, info, log_fn):
        """
        Normalize different Middlebury formats to:
        outdir/
          left.png  (or im0.png)
          right.png (or im1.png)
          disp0GT.pfm   (ground truth disparity)
          calib.txt
        """
        dtype = info["type"]

        if dtype == "middlebury_2001":
            # Typical structure: outdir/tsukuba/imL.png, imR.png
            imgs = list(outdir.rglob("imL.png")) + list(outdir.rglob("im2.ppm"))
            for src in imgs:
                dst = outdir / "left.png"
                if not dst.exists():
                    shutil.copy2(src, dst)

            imgs_r = list(outdir.rglob("imR.png")) + list(outdir.rglob("im6.ppm"))
            for src in imgs_r:
                dst = outdir / "right.png"
                if not dst.exists():
                    shutil.copy2(src, dst)

        elif dtype in ("middlebury_2003", "middlebury_2005"):
            # Look for im0/im1 pattern
            for pattern, dst_name in [("im0.png", "left.png"),
                                        ("im1.png", "right.png"),
                                        ("disp0.pfm", "disp0GT.pfm"),
                                        ("disp0GT.pfm", "disp0GT.pfm"),
                                        ("calib.txt", "calib.txt")]:
                found = list(outdir.rglob(pattern))
                if found:
                    dst = outdir / dst_name
                    if not dst.exists():
                        shutil.copy2(found[0], dst)

        log_fn(f"     Structure normalized.")

    # ----------------------------------------------------------
    def _is_complete(self, outdir, info):
        if not outdir.exists():
            return False
        meta = outdir / "meta.json"
        return meta.exists()

    def _save_meta(self, outdir, info):
        with open(outdir / "meta.json", "w") as f:
            json.dump(info, f, indent=2)

    # ----------------------------------------------------------
    # SYNTHETIC CALIBRATION IMAGES
    # ----------------------------------------------------------
    def generate_calibration_images(self, board_size=(9, 6),
                                    square_size=50,
                                    n_images=20,
                                    img_size=(640, 480),
                                    log_fn=print):
        """
        Generate synthetic chessboard calibration images with
        simulated camera distortion for both left and right cameras.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            log_fn("ERROR: OpenCV not installed.")
            return False

        log_fn("\n  🎲 Generating synthetic calibration images...")

        left_dir  = self.base_dir.parent / "calibration_images" / "left"
        right_dir = self.base_dir.parent / "calibration_images" / "right"
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)

        # Camera intrinsics (simulated)
        f  = 800.0
        cx = img_size[0] / 2
        cy = img_size[1] / 2
        K  = np.array([[f,   0, cx],
                        [0,   f, cy],
                        [0,   0,  1]], dtype=np.float64)

        # Distortion coefficients
        D_left  = np.array([ 0.1, -0.2, 0.001,  0.001, 0.05])
        D_right = np.array([ 0.12,-0.18, 0.002, -0.001, 0.04])

        # Stereo geometry
        T = np.array([[-0.12], [0.0], [0.0]])  # 12cm baseline

        board_w, board_h = board_size

        # 3D object points
        objp        = np.zeros((board_w * board_h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
        objp       *= square_size / 1000.0   # mm → meters

        generated = 0
        np.random.seed(42)

        for i in range(n_images):
            # Random board pose
            rvec = np.array([
                np.random.uniform(-0.4, 0.4),
                np.random.uniform(-0.4, 0.4),
                np.random.uniform(-0.3, 0.3)
            ], dtype=np.float64)

            tvec = np.array([
                np.random.uniform(-0.10, 0.10),
                np.random.uniform(-0.06, 0.06),
                np.random.uniform( 0.40, 0.80)
            ], dtype=np.float64)

            # Project to left camera
            pts_l, _ = cv2.projectPoints(objp, rvec, tvec, K, D_left)
            pts_l    = pts_l.reshape(-1, 2)

            # Project to right camera (shifted by baseline)
            tvec_r   = tvec.copy()
            tvec_r[0] -= 0.12
            pts_r, _ = cv2.projectPoints(objp, rvec, tvec_r, K, D_right)
            pts_r    = pts_r.reshape(-1, 2)

            # Check all points are in image
            def in_image(pts, w, h, margin=20):
                return (np.all(pts[:, 0] > margin) and
                        np.all(pts[:, 0] < w - margin) and
                        np.all(pts[:, 1] > margin) and
                        np.all(pts[:, 1] < h - margin))

            if not (in_image(pts_l, *img_size) and in_image(pts_r, *img_size)):
                continue

            # Draw chessboard on white image
            def draw_board(pts, size, bw, bh, sq):
                img = np.ones((*size[::-1], 3), dtype=np.uint8) * 230
                # Draw squares
                for row in range(bh):
                    for col in range(bw):
                        idx = row * bw + col
                        if (row + col) % 2 == 0:
                            # Get 4 corners of this square
                            c_tl = pts[idx]       if col < bw and row < bh else None
                            if idx < len(pts) - 1:
                                pass
                pts_int = pts.astype(np.int32)
                # Draw grid
                for r in range(bh):
                    for c in range(bw - 1):
                        p1 = tuple(pts_int[r * bw + c])
                        p2 = tuple(pts_int[r * bw + c + 1])
                        cv2.line(img, p1, p2, (100, 100, 100), 1)
                for r in range(bh - 1):
                    for c in range(bw):
                        p1 = tuple(pts_int[r * bw + c])
                        p2 = tuple(pts_int[(r+1) * bw + c])
                        cv2.line(img, p1, p2, (100, 100, 100), 1)
                # Draw corners
                for pt in pts_int:
                    cv2.circle(img, tuple(pt), 3, (0, 0, 255), -1)
                # Add noise
                noise = np.random.normal(0, 3, img.shape).astype(np.int16)
                img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                return img

            img_l = draw_board(pts_l, img_size, board_w, board_h, square_size)
            img_r = draw_board(pts_r, img_size, board_w, board_h, square_size)

            fname = f"calib_{i:02d}.png"
            cv2.imwrite(str(left_dir  / fname), img_l)
            cv2.imwrite(str(right_dir / fname), img_r)
            generated += 1
            log_fn(f"  ✓ Generated pair {generated}: {fname}")

        log_fn(f"\n  ✓ {generated} calibration pairs saved to:")
        log_fn(f"     {left_dir}")
        log_fn(f"     {right_dir}")
        return generated >= 5