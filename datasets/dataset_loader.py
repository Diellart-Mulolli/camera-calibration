# datasets/dataset_loader.py
# Scans dataset folder and provides structured access
# Student: Mërgim Pirraku | UBT 2026

import os
import cv2
import numpy as np
import json
from pathlib import Path


class DatasetLoader:
    """
    Scans the datasets/ folder and provides auto-detection
    of stereo image pairs and calibration data.
    """

    # Possible filename pairs in order of preference
    LEFT_NAMES  = ["left.png", "left.jpg", "im0.png",
                   "imL.png",  "image_left.png", "im0.jpg"]
    RIGHT_NAMES = ["right.png","right.jpg","im1.png",
                   "imR.png",  "image_right.png","im1.jpg"]
    GT_NAMES    = ["disp0GT.pfm","disp0.pfm","disp_left.pfm",
                   "groundtruth.pfm"]

    def __init__(self, datasets_dir=None):
        if datasets_dir is None:
            datasets_dir = Path(__file__).parent
        self.datasets_dir = Path(datasets_dir)

    # ----------------------------------------------------------
    def scan(self):
        """
        Returns list of dataset entries:
        [{'key': str, 'name': str, 'path': Path,
          'left': Path, 'right': Path,
          'has_gt': bool, 'meta': dict}]
        """
        entries = []

        if not self.datasets_dir.exists():
            return entries

        for folder in sorted(self.datasets_dir.iterdir()):
            if not folder.is_dir():
                continue
            if folder.name.startswith('.') or folder.name == '__pycache__':
                continue

            left  = self._find_image(folder, self.LEFT_NAMES)
            right = self._find_image(folder, self.RIGHT_NAMES)

            if left is None or right is None:
                # Try one level deeper
                for sub in folder.iterdir():
                    if sub.is_dir():
                        left  = self._find_image(sub, self.LEFT_NAMES)
                        right = self._find_image(sub, self.RIGHT_NAMES)
                        if left and right:
                            folder = sub
                            break

            if left is None or right is None:
                continue

            gt  = self._find_image(folder, self.GT_NAMES)
            meta_path = folder / "meta.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)

            # Read image size
            img = cv2.imread(str(left))
            h, w = (img.shape[:2] if img is not None else (0, 0))

            entries.append({
                'key':    folder.name,
                'name':   meta.get('name', folder.name.capitalize()),
                'path':   folder,
                'left':   left,
                'right':  right,
                'gt':     gt,
                'has_gt': gt is not None,
                'meta':   meta,
                'width':  w,
                'height': h,
                'description': meta.get('description', ''),
                'paper_section': meta.get('paper_section', ''),
            })

        return entries

    # ----------------------------------------------------------
    def load_pair(self, key_or_path):
        """Load left/right images for a dataset key or path."""
        entries = self.scan()

        entry = next((e for e in entries
                      if e['key'] == key_or_path), None)
        if entry is None:
            p = Path(key_or_path)
            entry = next((e for e in entries
                          if e['path'] == p), None)

        if entry is None:
            return None, None, None

        left  = cv2.imread(str(entry['left']))
        right = cv2.imread(str(entry['right']))
        gt    = self._load_pfm(str(entry['gt'])) if entry['has_gt'] else None

        return left, right, gt

    # ----------------------------------------------------------
    def _find_image(self, folder, names):
        for name in names:
            p = folder / name
            if p.exists():
                return p
        return None

    # ----------------------------------------------------------
    @staticmethod
    def _load_pfm(path):
        """Load Middlebury .pfm disparity file."""
        try:
            with open(path, 'rb') as f:
                header = f.readline().decode('utf-8').strip()
                color  = header == 'PF'

                dims   = f.readline().decode('utf-8').strip().split()
                w, h   = int(dims[0]), int(dims[1])

                scale  = float(f.readline().decode('utf-8').strip())
                endian = '<' if scale < 0 else '>'
                scale  = abs(scale)

                data = np.fromfile(f, endian + 'f')
                shape = (h, w, 3) if color else (h, w)
                data  = np.reshape(data, shape)
                data  = np.flipud(data)

                return data.astype(np.float32)
        except Exception:
            return None