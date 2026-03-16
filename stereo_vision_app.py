# stereo_vision_app.py
# Complete Stereo Vision Pipeline with GUI
# Paper: "Vlerësimi Gjeometrik i Thellësisë në Sistemet Stereo me Dy Kamera Statike"
# Student: Mërgim Pirraku | UBT 2026

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import os
import glob
from PIL import Image, ImageTk
import warnings
warnings.filterwarnings('ignore')

# ── 0. Dataset Browser ───────────────────────────────────
# Add to imports at top of stereo_vision_app.py:
from datasets.dataset_loader import DatasetLoader
from datasets.download_datasets import DatasetDownloader, DATASETS

# ── Add this method to StereoVisionApp class ─────────────

def _build_dataset_browser(self, parent):
    """Auto-scan datasets/ folder and show available datasets."""
    
    f0 = tk.Frame(parent, bg=self.PANEL, bd=0, relief='flat')
    f0.pack(fill='x', pady=(0, 6))

    tk.Label(f0, text="  📦  0. Dataset Browser",
             bg="#2d5a3d", fg='white',
             font=('Segoe UI', 10, 'bold'),
             anchor='w').pack(fill='x', ipady=4)

    inner = tk.Frame(f0, bg=self.PANEL)
    inner.pack(fill='x', padx=8, pady=6)

    # Scan button + refresh
    btn_row = tk.Frame(inner, bg=self.PANEL)
    btn_row.pack(fill='x', pady=(0, 4))

    self._make_btn(btn_row, "Scan Datasets",
                   self._scan_datasets, "🔍",
                   color="#2d5a3d", width=13
                   ).pack(side='left', padx=(0, 4))
    self._make_btn(btn_row, "Download",
                   self._open_download_window, "⬇",
                   color="#3a2d5a", width=10
                   ).pack(side='left')

    # Listbox showing found datasets
    list_frame = tk.Frame(inner, bg=self.PANEL)
    list_frame.pack(fill='x')

    sb = tk.Scrollbar(list_frame, bg=self.BTN_BG)
    sb.pack(side='right', fill='y')

    self.dataset_listbox = tk.Listbox(
        list_frame,
        bg=self.BTN_BG, fg=self.TEXT,
        font=('Consolas', 9),
        selectbackground=self.ACCENT,
        selectforeground='white',
        relief='flat', bd=0,
        height=6,
        yscrollcommand=sb.set
    )
    self.dataset_listbox.pack(fill='x')
    sb.config(command=self.dataset_listbox.yview)

    self.dataset_listbox.bind('<Double-Button-1>', self._load_selected_dataset)

    # Info label
    self.dataset_info = tk.Label(
        inner, text="Double-click to load",
        bg=self.PANEL, fg=self.TEXT_DIM,
        font=('Segoe UI', 8), wraplength=240
    )
    self.dataset_info.pack(fill='x', pady=(4, 0))

    # Load button
    self._make_btn(inner, "Load Selected Dataset",
                   self._load_selected_dataset, "▶",
                   color=self.ACCENT, width=26
                   ).pack(fill='x', pady=(4, 0))

    # Auto-scan on startup
    self.after(500, self._scan_datasets)

# ── Dataset scanning methods ─────────────────────────────

def _scan_datasets(self):
    """Scan datasets/ folder and populate listbox."""
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        datasets_dir = Path(__file__).parent / "datasets"

    loader  = DatasetLoader(datasets_dir)
    entries = loader.scan()

    self._dataset_entries = entries

    self.dataset_listbox.delete(0, 'end')

    if not entries:
        self.dataset_listbox.insert('end', "  No datasets found")
        self.dataset_listbox.insert('end', "  Click Download →")
        self.dataset_info.config(
            text="Use Download to get Middlebury datasets",
            fg=self.WARNING
        )
        return

    for e in entries:
        gt_flag = " [GT]" if e['has_gt'] else ""
        size    = f" {e['width']}×{e['height']}" if e['width'] else ""
        self.dataset_listbox.insert(
            'end',
            f"  {e['name']}{gt_flag}{size}"
        )

    self.dataset_info.config(
        text=f"Found {len(entries)} dataset(s). Double-click to load.",
        fg=self.SUCCESS
    )
    self._log(f"Scanned datasets/: found {len(entries)} datasets", "success")
    for e in entries:
        self._log(f"  ✓ {e['name']} — {e['description']}", "dim")

def _load_selected_dataset(self, event=None):
    """Load the selected dataset into the pipeline."""
    sel = self.dataset_listbox.curselection()
    if not sel:
        self._log("No dataset selected.", "warning")
        return

    idx   = sel[0]
    entry = self._dataset_entries[idx]

    self._log(f"\n── Loading dataset: {entry['name']} ──", "accent")

    left  = cv2.imread(str(entry['left']))
    right = cv2.imread(str(entry['right']))

    if left is None or right is None:
        self._log("ERROR: Could not read images.", "error")
        return

    self.pipeline.img_left  = left
    self.pipeline.img_right = right

    # Apply dataset calibration defaults if available
    meta = entry['meta']
    if 'focal_length' in meta:
        self.pipeline.focal_length = float(meta['focal_length'])
        self.focal_var.set(self.pipeline.focal_length)
    if 'baseline' in meta:
        self.pipeline.baseline = float(meta['baseline'])
        self.baseline_var.set(self.pipeline.baseline * 100)

    # Assume pre-rectified for Middlebury
    if 'middlebury' in meta.get('type', ''):
        self.pipeline.img_left_rect  = left
        self.pipeline.img_right_rect = right
        self._log("  Middlebury dataset: images are pre-rectified.", "warning")

    # Ground truth
    if entry['has_gt']:
        gt = DatasetLoader._load_pfm(str(entry['gt']))
        if gt is not None:
            self._log(f"  Ground truth loaded: {gt.shape}", "success")
            self.pipeline._gt_disparity = gt

    h, w = left.shape[:2]
    self._log(f"  Size: {w}×{h}  |  {entry['description']}", "dim")
    if entry['paper_section']:
        self._log(f"  Relevant sections: {entry['paper_section']}", "dim")

    self.img_status.config(
        text=f"Dataset: {entry['name']}",
        fg=self.SUCCESS
    )
    self.dataset_info.config(
        text=f"Loaded: {entry['name']} ({w}×{h})",
        fg=self.SUCCESS
    )

    self.after(100, lambda: self._display_on_canvas('left',  left))
    self.after(150, lambda: self._display_on_canvas('right', right))
    self.nb.select(0)
    self._log(f"✓ Dataset '{entry['name']}' loaded!", "success")

def _open_download_window(self):
    """Open download manager window."""
    win = tk.Toplevel(self)
    win.title("Dataset Download Manager")
    win.geometry("600x500")
    win.configure(bg=self.BG)

    tk.Label(win,
             text="  ⬇  Dataset Download Manager",
             bg=self.ACCENT, fg='white',
             font=('Segoe UI', 12, 'bold'),
             anchor='w').pack(fill='x', ipady=8)

    # Dataset list with checkboxes
    scroll_frame = tk.Frame(win, bg=self.BG)
    scroll_frame.pack(fill='both', expand=True, padx=10, pady=10)

    tk.Label(scroll_frame,
             text="Select datasets to download:",
             bg=self.BG, fg=self.TEXT,
             font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 6))

    check_vars = {}
    for key, info in DATASETS.items():
        var = tk.BooleanVar(value=(key in ["teddy", "tsukuba"]))
        check_vars[key] = var

        row = tk.Frame(scroll_frame, bg=self.PANEL)
        row.pack(fill='x', pady=2, padx=4)

        tk.Checkbutton(
            row, variable=var,
            bg=self.PANEL, fg=self.TEXT,
            selectcolor=self.BTN_BG,
            activebackground=self.PANEL,
            font=('Segoe UI', 9)
        ).pack(side='left')

        tk.Label(row, text=f"{info['name']}",
                 bg=self.PANEL, fg=self.ACCENT2,
                 font=('Segoe UI', 9, 'bold'),
                 width=18, anchor='w').pack(side='left')
        tk.Label(row, text=info['description'],
                 bg=self.PANEL, fg=self.TEXT_DIM,
                 font=('Segoe UI', 8)).pack(side='left')
        tk.Label(row, text=info['image_size'],
                 bg=self.PANEL, fg=self.INFO,
                 font=('Consolas', 8)).pack(side='right', padx=8)

    # Log area
    log = tk.Text(win, bg='#12121e', fg=self.TEXT,
                  font=('Consolas', 8), relief='flat',
                  height=8, state='disabled')
    log.pack(fill='x', padx=10, pady=(0, 4))

    def dl_log(msg, end="\n"):
        log.configure(state='normal')
        log.insert('end', str(msg) + end)
        log.see('end')
        log.configure(state='disabled')
        win.update_idletasks()

    def start_download():
        selected = [k for k, v in check_vars.items() if v.get()]
        if not selected:
            dl_log("Select at least one dataset.")
            return

        btn_dl.config(state='disabled', text="Downloading...")

        def task():
            datasets_dir = Path(__file__).parent / "datasets"
            downloader   = DatasetDownloader(datasets_dir)
            downloader.download_all(keys=selected, log_fn=dl_log)
            dl_log("\n✓ All downloads complete!")
            self.after(200, self._scan_datasets)
            btn_dl.config(state='normal', text="⬇  Download Selected")

        import threading
        threading.Thread(target=task, daemon=True).start()

    btn_dl = tk.Button(
        win, text="⬇  Download Selected",
        command=start_download,
        bg=self.ACCENT, fg='white',
        font=('Segoe UI', 10, 'bold'),
        relief='flat', pady=8, cursor='hand2'
    )
    btn_dl.pack(fill='x', padx=10, pady=6)
# ============================================================
# CORE ALGORITHMS (from paper)
# ============================================================

class StereoPipeline:
    """
    Implements all algorithms from the paper:
    - Zhang calibration (Section 4.3)
    - Rectification    (Section 4.4)
    - Disparity        (Section 5.7)
    - Depth Z = fB/d   (Section 5.8)
    - Error analysis   (Section 6)
    """

    def __init__(self):
        self.calib_data    = None
        self.rect_maps     = None
        self.img_left      = None
        self.img_right     = None
        self.img_left_rect = None
        self.img_right_rect= None
        self.disparity     = None
        self.depth_map     = None
        self.points_3d     = None

        # Default parameters (can be overridden from UI)
        self.focal_length  = 1735.0   # pixels
        self.baseline      = 0.160    # meters
        self.num_disp      = 128
        self.block_size    = 15
        self.sigma_d       = 0.5      # disparity uncertainty (pixels)

    # ----------------------------------------------------------
    # CALIBRATION - Section 4.3
    # ----------------------------------------------------------
    def calibrate_stereo(self, left_path, right_path,
                         board_size=(9, 6), square_size=0.025,
                         log_fn=print):

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0],
                                0:board_size[1]].T.reshape(-1, 2)
        objp *= square_size

        objpoints, imgpts_l, imgpts_r = [], [], []

        left_imgs  = sorted(glob.glob(os.path.join(left_path,  '*.png')) +
                            glob.glob(os.path.join(left_path,  '*.jpg')))
        right_imgs = sorted(glob.glob(os.path.join(right_path, '*.png')) +
                            glob.glob(os.path.join(right_path, '*.jpg')))

        if len(left_imgs) != len(right_imgs) or len(left_imgs) == 0:
            log_fn("ERROR: Image count mismatch or no images found.")
            return False

        img_shape = None
        valid = 0

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
                log_fn(f"  ✓ Pair {valid}: {os.path.basename(lp)}")
            else:
                log_fn(f"  ✗ Skipped: {os.path.basename(lp)}")

        if valid < 5:
            log_fn("ERROR: Need at least 5 valid pairs.")
            return False

        _, Kl, Dl, _, _ = cv2.calibrateCamera(objpoints, imgpts_l, img_shape, None, None)
        _, Kr, Dr, _, _ = cv2.calibrateCamera(objpoints, imgpts_r, img_shape, None, None)

        ret, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpts_l, imgpts_r,
            Kl, Dl, Kr, Dr, img_shape,
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=criteria
        )

        baseline = np.linalg.norm(T)

        self.calib_data = {
            'K_left': Kl, 'D_left': Dl,
            'K_right': Kr, 'D_right': Dr,
            'R': R, 'T': T, 'E': E, 'F': F,
            'baseline': baseline,
            'image_size': img_shape,
            'rms': ret
        }

        self.focal_length = float(Kl[0, 0])
        self.baseline     = float(baseline)

        log_fn(f"\n  Focal length  : {self.focal_length:.2f} px")
        log_fn(f"  Baseline B    : {self.baseline*100:.2f} cm")
        log_fn(f"  RMS error     : {ret:.4f} px")
        return True

    # ----------------------------------------------------------
    # RECTIFICATION - Section 4.4
    # ----------------------------------------------------------
    def compute_rectification(self, log_fn=print):
        if self.calib_data is None:
            log_fn("ERROR: Run calibration first.")
            return False

        c = self.calib_data
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            c['K_left'], c['D_left'],
            c['K_right'], c['D_right'],
            c['image_size'], c['R'], c['T'],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )

        sz = c['image_size']
        mLx, mLy = cv2.initUndistortRectifyMap(c['K_left'],  c['D_left'],  R1, P1, sz, cv2.CV_32FC1)
        mRx, mRy = cv2.initUndistortRectifyMap(c['K_right'], c['D_right'], R2, P2, sz, cv2.CV_32FC1)

        self.rect_maps = {
            'map_left_x': mLx,  'map_left_y': mLy,
            'map_right_x': mRx, 'map_right_y': mRy,
            'Q': Q, 'P1': P1,
            'focal_length_rect': float(P1[0, 0])
        }
        self.focal_length = float(P1[0, 0])
        log_fn(f"  Rectified focal length: {self.focal_length:.2f} px")
        return True

    def apply_rectification(self, log_fn=print):
        if self.rect_maps is None:
            log_fn("ERROR: Compute rectification maps first.")
            return False
        if self.img_left is None:
            log_fn("ERROR: Load images first.")
            return False

        m = self.rect_maps
        self.img_left_rect  = cv2.remap(self.img_left,  m['map_left_x'],  m['map_left_y'],  cv2.INTER_LANCZOS4)
        self.img_right_rect = cv2.remap(self.img_right, m['map_right_x'], m['map_right_y'], cv2.INTER_LANCZOS4)
        log_fn("  Rectification applied.")
        return True

    # ----------------------------------------------------------
    # DISPARITY - Section 5.7  d = xL - xR
    # ----------------------------------------------------------
    def compute_disparity(self, log_fn=print):
        left  = self.img_left_rect  if self.img_left_rect  is not None else self.img_left
        right = self.img_right_rect if self.img_right_rect is not None else self.img_right

        if left is None:
            log_fn("ERROR: Load images first.")
            return False

        gl = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY) if len(left.shape)  == 3 else left
        gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if len(right.shape) == 3 else right

        stereo = cv2.StereoBM_create(
            numDisparities=self.num_disp,
            blockSize=self.block_size
        )
        raw = stereo.compute(gl, gr).astype(np.float32) / 16.0

        # Filter invalid
        self.disparity = np.where(raw <= 0, np.nan, raw)

        valid_pct = 100 * np.sum(~np.isnan(self.disparity)) / self.disparity.size
        log_fn(f"  Valid disparity pixels: {valid_pct:.1f}%")
        log_fn(f"  Disparity range: [{np.nanmin(self.disparity):.1f}, {np.nanmax(self.disparity):.1f}] px")
        return True

    # ----------------------------------------------------------
    # DEPTH MAP - Section 5.8  Z = fB/d
    # ----------------------------------------------------------
    def compute_depth(self, log_fn=print):
        if self.disparity is None:
            log_fn("ERROR: Compute disparity first.")
            return False

        f, B = self.focal_length, self.baseline
        self.depth_map = np.where(
            np.isnan(self.disparity) | (self.disparity <= 0),
            np.nan,
            (f * B) / self.disparity
        )

        valid = self.depth_map[~np.isnan(self.depth_map)]
        if len(valid):
            log_fn(f"  f = {f:.2f} px,  B = {B*100:.2f} cm")
            log_fn(f"  Depth range  : [{np.nanmin(self.depth_map):.3f}, {np.nanmax(self.depth_map):.3f}] m")
            log_fn(f"  Mean depth   : {np.nanmean(self.depth_map):.3f} m")
            log_fn(f"  Median depth : {np.nanmedian(self.depth_map):.3f} m")
        return True

    # ----------------------------------------------------------
    # ERROR ANALYSIS - Section 6
    # ----------------------------------------------------------
    def get_error_analysis_data(self, Z_range=(0.3, 10.0), n=300):
        f, B, s = self.focal_length, self.baseline, self.sigma_d
        Z  = np.linspace(Z_range[0], Z_range[1], n)
        d  = (f * B) / Z
        sZ = (Z**2 / (f * B)) * s        # σ_Z = Z²/(fB) · σ_d
        rZ = sZ / Z * 100                  # relative error %
        sens = (f * B) / (d**2)           # |dZ/dd|
        return {'Z': Z, 'd': d, 'sigma_Z': sZ, 'rel_error': rZ, 'sensitivity': sens}

    def get_taylor_data(self, d0=None, delta_range=(-15, 15), n=300):
        if d0 is None:
            d0 = np.nanmedian(self.disparity) if self.disparity is not None else 30.0
        f, B = self.focal_length, self.baseline
        deltas  = np.linspace(delta_range[0], delta_range[1], n)
        Z_exact = (f * B) / np.maximum(d0 + deltas, 0.1)
        Z0      = (f * B) / d0
        slope   = -(f * B) / d0**2
        Z_lin   = Z0 + slope * deltas
        return {'deltas': deltas, 'Z_exact': Z_exact, 'Z_lin': Z_lin,
                'Z0': Z0, 'd0': d0, 'slope': slope}

    def get_depth_stats(self):
        if self.depth_map is None:
            return {}
        v = self.depth_map[~np.isnan(self.depth_map)]
        if len(v) == 0:
            return {}
        return {
            'min':    float(np.min(v)),
            'max':    float(np.max(v)),
            'mean':   float(np.mean(v)),
            'median': float(np.median(v)),
            'std':    float(np.std(v)),
            'valid':  100 * len(v) / self.depth_map.size
        }


# ============================================================
# MAIN APPLICATION
# ============================================================

class StereoVisionApp(tk.Tk):

    # color palette
    BG       = "#1e1e2e"
    PANEL    = "#2a2a3e"
    ACCENT   = "#7c6af7"
    ACCENT2  = "#56cfb2"
    BTN_BG   = "#3d3d5c"
    BTN_HOV  = "#5c5c8a"
    TEXT     = "#cdd6f4"
    TEXT_DIM = "#6c7086"
    SUCCESS  = "#a6e3a1"
    WARNING  = "#f9e2af"
    ERROR    = "#f38ba8"
    INFO     = "#89b4fa"

    def __init__(self):
        super().__init__()
        self.pipeline = StereoPipeline()

        self.title("Stereo Vision Depth Estimation  |  UBT 2026  |  Mërgim Pirraku")
        self.geometry("1550x920")
        self.configure(bg=self.BG)
        self.resizable(True, True)

        self._build_styles()
        self._build_ui()
        self._log("Welcome! Load stereo images or calibration data to begin.", "info")
        self._log("Paper: Vlerësimi Gjeometrik i Thellësisë në Sistemet Stereo me Dy Kamera Statike", "dim")

    # ----------------------------------------------------------
    # STYLES
    # ----------------------------------------------------------
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use('clam')

        s.configure('TFrame',      background=self.BG)
        s.configure('Panel.TFrame', background=self.PANEL)

        s.configure('TLabel',
                    background=self.BG, foreground=self.TEXT,
                    font=('Segoe UI', 10))
        s.configure('Header.TLabel',
                    background=self.BG, foreground=self.ACCENT,
                    font=('Segoe UI', 13, 'bold'))
        s.configure('Sub.TLabel',
                    background=self.PANEL, foreground=self.TEXT_DIM,
                    font=('Segoe UI', 9))
        s.configure('Stat.TLabel',
                    background=self.PANEL, foreground=self.TEXT,
                    font=('Segoe UI', 10))
        s.configure('StatVal.TLabel',
                    background=self.PANEL, foreground=self.ACCENT2,
                    font=('Segoe UI', 11, 'bold'))

        s.configure('TNotebook',       background=self.BG, borderwidth=0)
        s.configure('TNotebook.Tab',
                    background=self.BTN_BG, foreground=self.TEXT,
                    font=('Segoe UI', 10, 'bold'), padding=[14, 6])
        s.map('TNotebook.Tab',
              background=[('selected', self.ACCENT)],
              foreground=[('selected', '#ffffff')])

        s.configure('TScale',   background=self.PANEL, troughcolor=self.BTN_BG)
        s.configure('TSpinbox', fieldbackground=self.BTN_BG, foreground=self.TEXT)
        s.configure('Horizontal.TProgressbar',
                    troughcolor=self.BTN_BG,
                    background=self.ACCENT,
                    thickness=6)

    def _make_btn(self, parent, text, command, icon="", color=None, width=22):
        c = color or self.BTN_BG
        b = tk.Button(parent, text=f"  {icon}  {text}" if icon else f"  {text}",
                      command=command,
                      bg=c, fg=self.TEXT,
                      activebackground=self.BTN_HOV, activeforeground='white',
                      relief='flat', bd=0, cursor='hand2',
                      font=('Segoe UI', 10, 'bold'),
                      width=width, padx=6, pady=8)
        b.bind('<Enter>', lambda e: b.config(bg=self.BTN_HOV))
        b.bind('<Leave>', lambda e: b.config(bg=c))
        return b

    # ----------------------------------------------------------
    # UI LAYOUT
    # ----------------------------------------------------------
    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────
        top = tk.Frame(self, bg=self.ACCENT, height=48)
        top.pack(fill='x', side='top')
        top.pack_propagate(False)

        tk.Label(top,
                 text="  🎯  STEREO VISION DEPTH ESTIMATION",
                 bg=self.ACCENT, fg='white',
                 font=('Segoe UI', 14, 'bold')).pack(side='left', padx=20, pady=10)
        tk.Label(top,
                 text="UBT 2026  |  Mërgim Pirraku",
                 bg=self.ACCENT, fg='#e0d7ff',
                 font=('Segoe UI', 10)).pack(side='right', padx=20)

        # ── Progress bar ─────────────────────────────────────
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(self, variable=self.progress_var,
                                         style='Horizontal.TProgressbar',
                                         maximum=100)
        self.progress.pack(fill='x', side='top')

        # ── Main columns ─────────────────────────────────────
        main = tk.Frame(self, bg=self.BG)
        main.pack(fill='both', expand=True)

        left_col  = tk.Frame(main, bg=self.BG, width=290)
        left_col.pack(side='left', fill='y', padx=(8, 4), pady=8)
        left_col.pack_propagate(False)

        center_col = tk.Frame(main, bg=self.BG)
        center_col.pack(side='left', fill='both', expand=True, padx=4, pady=8)

        right_col  = tk.Frame(main, bg=self.BG, width=310)
        right_col.pack(side='right', fill='y', padx=(4, 8), pady=8)
        right_col.pack_propagate(False)
        
        self._build_left_panel(left_col)
        self._build_center_panel(center_col)
        self._build_right_panel(right_col)

    # ----------------------------------------------------------
    # LEFT PANEL  — controls
    # ----------------------------------------------------------
    def _build_left_panel(self, parent):

        # ── Section helper ───────────────────────────────────
        def section(title):
            f = tk.Frame(parent, bg=self.PANEL, bd=0, relief='flat')
            f.pack(fill='x', pady=(0, 6))
            tk.Label(f, text=f"  {title}",
                     bg=self.ACCENT, fg='white',
                     font=('Segoe UI', 10, 'bold'),
                     anchor='w').pack(fill='x', ipady=4)
            inner = tk.Frame(f, bg=self.PANEL)
            inner.pack(fill='x', padx=8, pady=6)
            return inner
        
        # ── 1. Load images ───────────────────────────────────
        f1 = section("📂  1. Load Images")
        self._make_btn(f1, "Load Left Image",  self._load_left,  "🖼", width=26).pack(fill='x', pady=2)
        self._make_btn(f1, "Load Right Image", self._load_right, "🖼", width=26).pack(fill='x', pady=2)
        self._make_btn(f1, "Load Stereo Pair (folder)", self._load_folder, "📁", width=26).pack(fill='x', pady=2)

        self.img_status = tk.Label(f1, text="No images loaded",
                                   bg=self.PANEL, fg=self.TEXT_DIM,
                                   font=('Segoe UI', 9), wraplength=240)
        self.img_status.pack(fill='x', pady=(4, 0))

        # ── 2. Calibration ───────────────────────────────────
        f2 = section("🔧  2. Calibration  (§4.3)")
        self._make_btn(f2, "Run Stereo Calibration", self._run_calibration, "⚙", width=26).pack(fill='x', pady=2)
        self._make_btn(f2, "Load Saved Calibration", self._load_calibration, "💾", width=26).pack(fill='x', pady=2)
        self._make_btn(f2, "Use Dataset Defaults",   self._use_defaults,    "✔", width=26,
                       color="#2d5a3d").pack(fill='x', pady=2)

        # Board size
        bf = tk.Frame(f2, bg=self.PANEL)
        bf.pack(fill='x', pady=(4,0))
        tk.Label(bf, text="Board (cols × rows):", bg=self.PANEL,
                 fg=self.TEXT_DIM, font=('Segoe UI',9)).pack(anchor='w')
        brow = tk.Frame(bf, bg=self.PANEL)
        brow.pack(fill='x')
        self.board_cols = tk.IntVar(value=9)
        self.board_rows = tk.IntVar(value=6)
        tk.Spinbox(brow, from_=4, to=20, textvariable=self.board_cols, width=4,
                   bg=self.BTN_BG, fg=self.TEXT, bd=0,
                   buttonbackground=self.BTN_BG).pack(side='left', padx=(0,4))
        tk.Label(brow, text="×", bg=self.PANEL, fg=self.TEXT).pack(side='left')
        tk.Spinbox(brow, from_=4, to=20, textvariable=self.board_rows, width=4,
                   bg=self.BTN_BG, fg=self.TEXT, bd=0,
                   buttonbackground=self.BTN_BG).pack(side='left', padx=(4,0))

        self.calib_status = tk.Label(f2, text="Not calibrated",
                                     bg=self.PANEL, fg=self.TEXT_DIM,
                                     font=('Segoe UI', 9), wraplength=240)
        self.calib_status.pack(fill='x', pady=(4,0))

        # ── 3. Algorithm params ──────────────────────────────
        f3 = section("⚙  3. Algorithm Parameters")

        def slider_row(parent, label, var, from_, to, fmt="{:.0f}"):
            row = tk.Frame(parent, bg=self.PANEL)
            row.pack(fill='x', pady=3)
            lbl = tk.Label(row, text=label, bg=self.PANEL,
                           fg=self.TEXT_DIM, font=('Segoe UI',9), width=14, anchor='w')
            lbl.pack(side='left')
            val_lbl = tk.Label(row, text=fmt.format(var.get()),
                               bg=self.PANEL, fg=self.ACCENT2,
                               font=('Segoe UI', 9, 'bold'), width=7)
            val_lbl.pack(side='right')
            def on_change(v, vl=val_lbl, f=fmt, vv=var):
                vl.config(text=f.format(float(v)))
            s = ttk.Scale(row, from_=from_, to=to, variable=var,
                          orient='horizontal', command=on_change)
            s.pack(side='left', fill='x', expand=True, padx=4)

        self.num_disp_var  = tk.IntVar(value=128)
        self.block_var     = tk.IntVar(value=15)
        self.sigma_d_var   = tk.DoubleVar(value=0.5)
        self.focal_var     = tk.DoubleVar(value=1735.0)
        self.baseline_var  = tk.DoubleVar(value=16.0)   # cm

        slider_row(f3, "Num Disparities", self.num_disp_var,  16, 256)
        slider_row(f3, "Block Size",      self.block_var,      5,  51)
        slider_row(f3, "σ_d (pixels)",    self.sigma_d_var,  0.1, 3.0, "{:.2f}")
        slider_row(f3, "Focal f (px)",    self.focal_var,    200, 4000, "{:.0f}")
        slider_row(f3, "Baseline (cm)",   self.baseline_var,   1,  50, "{:.1f}")

        # ── 4. Pipeline buttons ──────────────────────────────
        f4 = section("▶  4. Run Pipeline")
        self._make_btn(f4, "Rectify Images",    self._run_rectification, "↔", width=26,
                       color="#3a3a6a").pack(fill='x', pady=2)
        self._make_btn(f4, "Compute Disparity", self._run_disparity,     "📐", width=26,
                       color="#3a3a6a").pack(fill='x', pady=2)
        self._make_btn(f4, "Compute Depth Map", self._run_depth,         "📏", width=26,
                       color="#3a3a6a").pack(fill='x', pady=2)
        self._make_btn(f4, "▶▶  Run Full Pipeline", self._run_full_pipeline, "🚀", width=26,
                       color=self.ACCENT).pack(fill='x', pady=(6, 2))

        # ── 5. Analysis ──────────────────────────────────────
        f5 = section("📊  5. Analysis & Export")
        self._make_btn(f5, "Error Analysis Plots", self._show_error_analysis, "📈", width=26).pack(fill='x', pady=2)
        self._make_btn(f5, "Taylor Linearization", self._show_taylor,         "∂",  width=26).pack(fill='x', pady=2)
        self._make_btn(f5, "Depth Histogram",      self._show_histogram,      "📊", width=26).pack(fill='x', pady=2)
        self._make_btn(f5, "Save Results",         self._save_results,        "💾", width=26).pack(fill='x', pady=2)

    # ----------------------------------------------------------
    # CENTER PANEL  — image / plot notebook
    # ----------------------------------------------------------
    def _build_center_panel(self, parent):
        self.nb = ttk.Notebook(parent)
        self.nb.pack(fill='both', expand=True)

        tabs = [
            ("Left Image",    self._tab_left),
            ("Right Image",   self._tab_right),
            ("Rectified",     self._tab_rect),
            ("Disparity Map", self._tab_disp),
            ("Depth Map",     self._tab_depth),
            ("3D Analysis",   self._tab_3d),
        ]

        self.canvases = {}
        self.tab_frames = {}

        for name, _ in tabs:
            frame = ttk.Frame(self.nb, style='TFrame')
            self.nb.add(frame, text=f"  {name}  ")
            self.tab_frames[name] = frame

        # Create placeholder figures
        for name, init_fn in tabs:
            init_fn(self.tab_frames[name])

    def _make_image_canvas(self, parent, name):
        """Canvas that fills its tab"""
        canvas = tk.Canvas(parent, bg='#12121e', highlightthickness=0)
        canvas.pack(fill='both', expand=True)
        self.canvases[name] = canvas
        canvas.bind('<Configure>', lambda e, n=name: self._resize_canvas(n))
        return canvas

    def _tab_left(self, f):
        tk.Label(f, text="Load left image →", bg='#12121e', fg=self.TEXT_DIM,
                 font=('Segoe UI', 12)).pack(expand=True)
        self._make_image_canvas(f, 'left')

    def _tab_right(self, f):
        tk.Label(f, text="Load right image →", bg='#12121e', fg=self.TEXT_DIM,
                 font=('Segoe UI', 12)).pack(expand=True)
        self._make_image_canvas(f, 'right')

    def _tab_rect(self, f):
        tk.Label(f, text="Run rectification →", bg='#12121e', fg=self.TEXT_DIM,
                 font=('Segoe UI', 12)).pack(expand=True)
        self._make_image_canvas(f, 'rect')

    def _tab_disp(self, f):
        tk.Label(f, text="Compute disparity →", bg='#12121e', fg=self.TEXT_DIM,
                 font=('Segoe UI', 12)).pack(expand=True)
        self._make_image_canvas(f, 'disp')

    def _tab_depth(self, f):
        tk.Label(f, text="Compute depth map →", bg='#12121e', fg=self.TEXT_DIM,
                 font=('Segoe UI', 12)).pack(expand=True)
        self._make_image_canvas(f, 'depth')

    def _tab_3d(self, f):
        # Matplotlib figure embedded for 3D analysis
        self.fig_3d = Figure(figsize=(8, 5), dpi=96,
                             facecolor='#12121e')
        self.ax_3d  = self.fig_3d.add_subplot(111)
        self.ax_3d.set_facecolor('#12121e')
        self.ax_3d.text(0.5, 0.5, 'Run analysis to see plots',
                        ha='center', va='center', color=self.TEXT_DIM,
                        fontsize=13, transform=self.ax_3d.transAxes)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=f)
        self.canvas_3d.get_tk_widget().pack(fill='both', expand=True)

    # ----------------------------------------------------------
    # RIGHT PANEL  — statistics + log
    # ----------------------------------------------------------
    def _build_right_panel(self, parent):

        # ── Statistics card ──────────────────────────────────
        stat_frame = tk.Frame(parent, bg=self.PANEL, bd=0)
        stat_frame.pack(fill='x', pady=(0, 6))

        tk.Label(stat_frame, text="  📊  Statistics",
                 bg=self.ACCENT, fg='white',
                 font=('Segoe UI', 10, 'bold'), anchor='w').pack(fill='x', ipady=4)

        self.stat_inner = tk.Frame(stat_frame, bg=self.PANEL)
        self.stat_inner.pack(fill='x', padx=10, pady=8)

        self._stat_vars = {}
        stat_defs = [
            ("f (focal length)",  "focal",   "px"),
            ("B (baseline)",      "baseline","cm"),
            ("Min depth",         "min",     "m"),
            ("Max depth",         "max",     "m"),
            ("Mean depth",        "mean",    "m"),
            ("Median depth",      "median",  "m"),
            ("Std deviation",     "std",     "m"),
            ("Valid pixels",      "valid",   "%"),
            ("σ_Z @ median",      "sigma_z", "cm"),
            ("Rel. error @ med.", "rel_err", "%"),
        ]

        for label, key, unit in stat_defs:
            row = tk.Frame(self.stat_inner, bg=self.PANEL)
            row.pack(fill='x', pady=2)
            tk.Label(row, text=label + ":", bg=self.PANEL,
                     fg=self.TEXT_DIM, font=('Segoe UI', 9),
                     anchor='w', width=18).pack(side='left')
            var = tk.StringVar(value="—")
            self._stat_vars[key] = var
            tk.Label(row, textvariable=var, bg=self.PANEL,
                     fg=self.ACCENT2, font=('Segoe UI', 10, 'bold'),
                     anchor='e').pack(side='left', padx=(4,0))
            tk.Label(row, text=unit, bg=self.PANEL,
                     fg=self.TEXT_DIM, font=('Segoe UI', 9)).pack(side='left', padx=(2,0))

        ttk.Separator(parent).pack(fill='x', pady=4)

        # ── Paper formulas reference ─────────────────────────
        ref_frame = tk.Frame(parent, bg=self.PANEL)
        ref_frame.pack(fill='x', pady=(0,6))

        tk.Label(ref_frame, text="  📐  Key Formulas",
                 bg=self.ACCENT, fg='white',
                 font=('Segoe UI', 10, 'bold'), anchor='w').pack(fill='x', ipady=4)

        formulas = [
            ("Depth (§5.8)",        "Z = f·B / d"),
            ("Disparity (§5.7)",    "d = xL − xR"),
            ("Error (§6.3)",        "σ_Z = (Z²/fB)·σ_d"),
            ("Sensitivity (§6.2)",  "dZ/dd = −fB/d²"),
            ("Relative err (§6.3)", "σ_Z/Z = (Z/fB)·σ_d"),
        ]

        for name, formula in formulas:
            row = tk.Frame(ref_frame, bg=self.PANEL)
            row.pack(fill='x', padx=10, pady=2)
            tk.Label(row, text=name, bg=self.PANEL, fg=self.TEXT_DIM,
                     font=('Segoe UI', 8), width=18, anchor='w').pack(side='left')
            tk.Label(row, text=formula, bg=self.PANEL, fg=self.INFO,
                     font=('Consolas', 9, 'bold')).pack(side='left')

        ttk.Separator(parent).pack(fill='x', pady=4)

        # ── Log console ──────────────────────────────────────
        log_frame = tk.Frame(parent, bg=self.PANEL)
        log_frame.pack(fill='both', expand=True)

        tk.Label(log_frame, text="  📋  Console Log",
                 bg=self.ACCENT, fg='white',
                 font=('Segoe UI', 10, 'bold'), anchor='w').pack(fill='x', ipady=4)

        self.log_text = tk.Text(
            log_frame, bg='#12121e', fg=self.TEXT,
            font=('Consolas', 9), relief='flat', bd=0,
            wrap='word', state='disabled', height=18
        )
        sb = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.log_text.pack(fill='both', expand=True, padx=2, pady=2)

        # Tag colors for log
        self.log_text.tag_config('info',    foreground=self.INFO)
        self.log_text.tag_config('success', foreground=self.SUCCESS)
        self.log_text.tag_config('warning', foreground=self.WARNING)
        self.log_text.tag_config('error',   foreground=self.ERROR)
        self.log_text.tag_config('dim',     foreground=self.TEXT_DIM)
        self.log_text.tag_config('accent',  foreground=self.ACCENT2)

        # Clear log button
        self._make_btn(log_frame, "Clear Log", lambda: self._clear_log(), width=14
                       ).pack(pady=4)

    # ----------------------------------------------------------
    # LOGGING
    # ----------------------------------------------------------
    def _log(self, msg, tag='info'):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', msg + "\n", tag)
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _clear_log(self):
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state='disabled')

    def _set_progress(self, val):
        self.progress_var.set(val)
        self.update_idletasks()

    # ----------------------------------------------------------
    # IMAGE DISPLAY
    # ----------------------------------------------------------
    def _np_to_photoimage(self, img_np, target_w, target_h):
        """Convert numpy array to Tkinter PhotoImage, preserving aspect ratio"""
        if img_np is None:
            return None

        h, w = img_np.shape[:2]
        scale = min(target_w / w, target_h / h) * 0.96
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(img_np, (nw, nh), interpolation=cv2.INTER_AREA)

        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        return ImageTk.PhotoImage(Image.fromarray(resized))

    def _display_on_canvas(self, canvas_key, img_np):
        canvas = self.canvases.get(canvas_key)
        if canvas is None or img_np is None:
            return

        w = canvas.winfo_width()  or 800
        h = canvas.winfo_height() or 500

        photo = self._np_to_photoimage(img_np, w, h)
        if photo is None:
            return

        canvas.delete('all')
        canvas.create_image(w // 2, h // 2, anchor='center', image=photo)
        canvas._photo = photo   # keep reference!

    def _resize_canvas(self, name):
        img_map = {
            'left':  self.pipeline.img_left,
            'right': self.pipeline.img_right,
            'disp':  self._disparity_as_colormap(),
            'depth': self._depth_as_colormap(),
        }
        if name == 'rect' and self.pipeline.img_left_rect is not None:
            combined = self._rectified_combined()
            self._display_on_canvas('rect', combined)
        elif name in img_map and img_map[name] is not None:
            self._display_on_canvas(name, img_map[name])

    def _disparity_as_colormap(self):
        if self.pipeline.disparity is None:
            return None
        d = np.nan_to_num(self.pipeline.disparity, nan=0).astype(np.float32)
        d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)

    def _depth_as_colormap(self):
        if self.pipeline.depth_map is None:
            return None
        d = np.nan_to_num(self.pipeline.depth_map, nan=0).astype(np.float32)
        p95 = np.percentile(d[d > 0], 95) if np.any(d > 0) else 1.0
        d = np.clip(d, 0, p95)
        d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)

    def _rectified_combined(self):
        if self.pipeline.img_left_rect is None:
            return None
        l = self.pipeline.img_left_rect
        r = self.pipeline.img_right_rect
        combined = np.hstack([l, r])
        if len(combined.shape) == 2:
            combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        h = combined.shape[0]
        for i in range(0, h, h // 12):
            combined[i:i+2, :] = [0, 200, 80]
        return combined

    # ----------------------------------------------------------
    # STATISTICS UPDATE
    # ----------------------------------------------------------
    def _update_stats(self):
        p = self.pipeline
        stats = p.get_depth_stats()

        self._stat_vars['focal'].set(f"{p.focal_length:.1f}")
        self._stat_vars['baseline'].set(f"{p.baseline*100:.2f}")

        if stats:
            self._stat_vars['min'].set(f"{stats['min']:.3f}")
            self._stat_vars['max'].set(f"{stats['max']:.3f}")
            self._stat_vars['mean'].set(f"{stats['mean']:.3f}")
            self._stat_vars['median'].set(f"{stats['median']:.3f}")
            self._stat_vars['std'].set(f"{stats['std']:.3f}")
            self._stat_vars['valid'].set(f"{stats['valid']:.1f}")

            Z_med  = stats['median']
            sigma_z = (Z_med**2 / (p.focal_length * p.baseline)) * p.sigma_d
            rel_err = (Z_med / (p.focal_length * p.baseline)) * p.sigma_d * 100

            self._stat_vars['sigma_z'].set(f"{sigma_z*100:.3f}")
            self._stat_vars['rel_err'].set(f"{rel_err:.3f}")

    # ----------------------------------------------------------
    # BUTTON HANDLERS
    # ----------------------------------------------------------
    def _load_left(self):
        path = filedialog.askopenfilename(
            title="Select Left Image",
            filetypes=[("Images", "*.png *.jpg *.bmp *.tiff"), ("All", "*.*")]
        )
        if path:
            self.pipeline.img_left = cv2.imread(path)
            self._log(f"Left image: {os.path.basename(path)}", "success")
            self.img_status.config(text=f"L: {os.path.basename(path)}", fg=self.SUCCESS)
            self.after(100, lambda: self._display_on_canvas('left', self.pipeline.img_left))
            self.nb.select(0)

    def _load_right(self):
        path = filedialog.askopenfilename(
            title="Select Right Image",
            filetypes=[("Images", "*.png *.jpg *.bmp *.tiff"), ("All", "*.*")]
        )
        if path:
            self.pipeline.img_right = cv2.imread(path)
            self._log(f"Right image: {os.path.basename(path)}", "success")
            self.img_status.config(text=f"R: {os.path.basename(path)}", fg=self.SUCCESS)
            self.after(100, lambda: self._display_on_canvas('right', self.pipeline.img_right))
            self.nb.select(1)

    def _load_folder(self):
        folder = filedialog.askdirectory(title="Select folder with left.png & right.png")
        if folder:
            lp = os.path.join(folder, 'left.png')
            rp = os.path.join(folder, 'right.png')
            if not os.path.exists(lp):
                lp = os.path.join(folder, 'im0.png')
                rp = os.path.join(folder, 'im1.png')
            if os.path.exists(lp) and os.path.exists(rp):
                self.pipeline.img_left  = cv2.imread(lp)
                self.pipeline.img_right = cv2.imread(rp)
                self._log(f"Loaded stereo pair from: {os.path.basename(folder)}", "success")
                h, w = self.pipeline.img_left.shape[:2]
                self._log(f"  Image size: {w}×{h} px", "dim")
                self.img_status.config(text=f"Pair loaded: {os.path.basename(folder)}", fg=self.SUCCESS)
                self.after(100, lambda: self._display_on_canvas('left',  self.pipeline.img_left))
                self.after(150, lambda: self._display_on_canvas('right', self.pipeline.img_right))
            else:
                self._log("Could not find left/right images in folder.", "error")

    def _run_calibration(self):
        ldir = filedialog.askdirectory(title="Select LEFT calibration images folder")
        if not ldir:
            return
        rdir = filedialog.askdirectory(title="Select RIGHT calibration images folder")
        if not rdir:
            return

        def task():
            self._log("\n── Stereo Calibration (Zhang's Method §4.3) ──", "accent")
            self._set_progress(10)
            ok = self.pipeline.calibrate_stereo(
                ldir, rdir,
                board_size=(self.board_cols.get(), self.board_rows.get()),
                log_fn=self._log
            )
            self._set_progress(60)
            if ok:
                ok2 = self.pipeline.compute_rectification(self._log)
                self._set_progress(90)
                if ok2:
                    self.calib_status.config(
                        text=f"✓ Calibrated  RMS={self.pipeline.calib_data['rms']:.3f} px",
                        fg=self.SUCCESS)
                    self._log("Calibration complete!", "success")
                    self._update_stats()
            else:
                self.calib_status.config(text="✗ Calibration failed", fg=self.ERROR)
            self._set_progress(100)
            self.after(1500, lambda: self._set_progress(0))

        threading.Thread(target=task, daemon=True).start()

    def _load_calibration(self):
        path = filedialog.askopenfilename(
            title="Load calibration file",
            filetypes=[("NumPy archive", "*.npz"), ("All", "*.*")]
        )
        if path:
            try:
                data = np.load(path)
                self.pipeline.calib_data = {k: data[k] for k in data.files}
                self.pipeline.baseline     = float(data['baseline'])
                self.pipeline.focal_length = float(data['K_left'][0, 0])
                self.pipeline.compute_rectification(self._log)
                self.calib_status.config(text=f"✓ Loaded: {os.path.basename(path)}", fg=self.SUCCESS)
                self._log(f"Calibration loaded from {os.path.basename(path)}", "success")
                self._update_stats()
            except Exception as e:
                self._log(f"Failed to load calibration: {e}", "error")

    def _use_defaults(self):
        """Use Middlebury dataset default parameters"""
        self.pipeline.focal_length = 1735.0
        self.pipeline.baseline     = 0.160
        self.focal_var.set(1735.0)
        self.baseline_var.set(16.0)
        self.calib_status.config(text="✓ Using Middlebury defaults", fg=self.WARNING)
        self._log("Using Middlebury dataset defaults: f=1735 px, B=16 cm", "warning")
        self._update_stats()

    def _sync_params(self):
        """Sync UI sliders → pipeline"""
        nd = int(self.num_disp_var.get())
        nd = nd if nd % 16 == 0 else (nd // 16) * 16
        self.pipeline.num_disp     = max(16, nd)
        self.pipeline.block_size   = int(self.block_var.get())
        if self.pipeline.block_size % 2 == 0:
            self.pipeline.block_size += 1
        self.pipeline.sigma_d      = float(self.sigma_d_var.get())
        self.pipeline.focal_length = float(self.focal_var.get())
        self.pipeline.baseline     = float(self.baseline_var.get()) / 100.0

    def _run_rectification(self):
        def task():
            self._log("\n── Rectification (§4.4) ──", "accent")
            self._set_progress(20)
            if self.pipeline.calib_data:
                ok = self.pipeline.apply_rectification(self._log)
            else:
                # No calibration → images are already rectified (dataset)
                self.pipeline.img_left_rect  = self.pipeline.img_left
                self.pipeline.img_right_rect = self.pipeline.img_right
                ok = True
                self._log("  No calibration data: assuming pre-rectified images.", "warning")

            self._set_progress(80)
            if ok:
                combined = self._rectified_combined()
                self.after(100, lambda: self._display_on_canvas('rect', combined))
                self.nb.select(2)
                self._log("Rectification applied (green lines = epipolar lines)", "success")
            self._set_progress(100)
            self.after(1500, lambda: self._set_progress(0))

        threading.Thread(target=task, daemon=True).start()

    def _run_disparity(self):
        self._sync_params()

        def task():
            self._log("\n── Disparity Map  d = xL − xR  (§5.7) ──", "accent")
            self._set_progress(15)
            ok = self.pipeline.compute_disparity(self._log)
            self._set_progress(85)
            if ok:
                img = self._disparity_as_colormap()
                self.after(100, lambda: self._display_on_canvas('disp', img))
                self.nb.select(3)
                self._log("Disparity map computed (plasma colormap: bright=near)", "success")
            self._set_progress(100)
            self.after(1500, lambda: self._set_progress(0))

        threading.Thread(target=task, daemon=True).start()

    def _run_depth(self):
        self._sync_params()

        def task():
            self._log("\n── Depth Map  Z = f·B/d  (§5.8) ──", "accent")
            self._set_progress(15)
            ok = self.pipeline.compute_depth(self._log)
            self._set_progress(85)
            if ok:
                img = self._depth_as_colormap()
                self.after(100, lambda: self._display_on_canvas('depth', img))
                self.nb.select(4)
                self._update_stats()
                self._log("Depth map computed (inferno: bright=far)", "success")
            self._set_progress(100)
            self.after(1500, lambda: self._set_progress(0))

        threading.Thread(target=task, daemon=True).start()

    def _run_full_pipeline(self):
        self._sync_params()

        def task():
            self._log("\n══ FULL PIPELINE ══", "accent")
            self._set_progress(5)

            if self.pipeline.img_left is None:
                self._log("ERROR: Load images first!", "error")
                return

            # Step 1: Rectify
            self._log("Step 1/3 — Rectification (§4.4)...", "info")
            self._set_progress(15)
            if self.pipeline.calib_data:
                self.pipeline.apply_rectification(self._log)
            else:
                self.pipeline.img_left_rect  = self.pipeline.img_left
                self.pipeline.img_right_rect = self.pipeline.img_right
                self._log("  Using pre-rectified images.", "warning")

            combined = self._rectified_combined()
            self.after(100, lambda: self._display_on_canvas('rect', combined))
            self._set_progress(33)

            # Step 2: Disparity
            self._log("Step 2/3 — Disparity (§5.7)...", "info")
            self.pipeline.compute_disparity(self._log)
            img_d = self._disparity_as_colormap()
            self.after(100, lambda: self._display_on_canvas('disp', img_d))
            self._set_progress(66)

            # Step 3: Depth
            self._log("Step 3/3 — Depth Z = f·B/d (§5.8)...", "info")
            self.pipeline.compute_depth(self._log)
            img_z = self._depth_as_colormap()
            self.after(100, lambda: self._display_on_canvas('depth', img_z))
            self._set_progress(90)

            self._update_stats()
            self._log("\n✓ Pipeline complete!", "success")

            # Auto-show depth tab
            self.nb.select(4)
            self._set_progress(100)
            self.after(1500, lambda: self._set_progress(0))

        threading.Thread(target=task, daemon=True).start()

    # ----------------------------------------------------------
    # ANALYSIS PLOTS
    # ----------------------------------------------------------
    def _show_error_analysis(self):
        self._sync_params()
        p = self.pipeline

        self._log("\n── Error Analysis (§6.3) σ_Z = (Z²/fB)·σ_d ──", "accent")

        data = p.get_error_analysis_data()
        Z, sZ, rZ, sens = data['Z'], data['sigma_Z'], data['rel_error'], data['sensitivity']

        self.fig_3d.clear()
        self.fig_3d.patch.set_facecolor('#12121e')

        axes = self.fig_3d.subplots(1, 3)
        colors_bg = '#12121e'

        # Absolute error
        axes[0].set_facecolor(colors_bg)
        axes[0].plot(Z, sZ * 100, color='#f38ba8', linewidth=2.5)
        axes[0].fill_between(Z, 0, sZ * 100, alpha=0.2, color='#f38ba8')
        axes[0].set_xlabel('Depth Z (m)', color=self.TEXT, fontsize=9)
        axes[0].set_ylabel('σ_Z (cm)', color=self.TEXT, fontsize=9)
        axes[0].set_title(f'Absolute Error\nσ_Z = Z²/(fB)·σ_d', color=self.ACCENT2, fontsize=9)
        axes[0].tick_params(colors=self.TEXT_DIM, labelsize=8)
        axes[0].spines[:].set_color(self.BTN_BG)
        axes[0].grid(True, alpha=0.15, color=self.TEXT_DIM)

        # Relative error
        axes[1].set_facecolor(colors_bg)
        axes[1].plot(Z, rZ, color='#89b4fa', linewidth=2.5)
        axes[1].fill_between(Z, 0, rZ, alpha=0.2, color='#89b4fa')
        axes[1].set_xlabel('Depth Z (m)', color=self.TEXT, fontsize=9)
        axes[1].set_ylabel('σ_Z/Z (%)', color=self.TEXT, fontsize=9)
        axes[1].set_title('Relative Error\nσ_Z/Z = (Z/fB)·σ_d', color=self.ACCENT2, fontsize=9)
        axes[1].tick_params(colors=self.TEXT_DIM, labelsize=8)
        axes[1].spines[:].set_color(self.BTN_BG)
        axes[1].grid(True, alpha=0.15, color=self.TEXT_DIM)

        # Baseline comparison
        axes[2].set_facecolor(colors_bg)
        baselines = [0.05, 0.10, 0.20, 0.30]
        palette   = ['#f38ba8', '#f9e2af', '#a6e3a1', '#89b4fa']
        for b, c in zip(baselines, palette):
            sZ_b = (Z**2 / (p.focal_length * b)) * p.sigma_d
            axes[2].plot(Z, sZ_b * 100, color=c, linewidth=2,
                         label=f'B={b*100:.0f}cm')
        axes[2].set_xlabel('Depth Z (m)', color=self.TEXT, fontsize=9)
        axes[2].set_ylabel('σ_Z (cm)', color=self.TEXT, fontsize=9)
        axes[2].set_title('Effect of Baseline B\n(larger B = more accurate)',
                           color=self.ACCENT2, fontsize=9)
        axes[2].tick_params(colors=self.TEXT_DIM, labelsize=8)
        axes[2].spines[:].set_color(self.BTN_BG)
        axes[2].grid(True, alpha=0.15, color=self.TEXT_DIM)
        axes[2].legend(fontsize=8, facecolor=self.PANEL, labelcolor=self.TEXT)

        self.fig_3d.suptitle(
            f'Error Analysis  (f={p.focal_length:.0f}px, B={p.baseline*100:.1f}cm, σ_d={p.sigma_d:.2f}px)',
            color=self.TEXT, fontsize=10, y=1.01)
        self.fig_3d.tight_layout()
        self.canvas_3d.draw()
        self.nb.select(5)
        self._log("Error analysis plots updated.", "success")

    def _show_taylor(self):
        self._sync_params()
        p = self.pipeline

        self._log("\n── Taylor Linearization (§6.2) ──", "accent")
        data = p.get_taylor_data()

        self.fig_3d.clear()
        self.fig_3d.patch.set_facecolor('#12121e')
        axes = self.fig_3d.subplots(1, 2)

        for ax in axes:
            ax.set_facecolor('#12121e')
            ax.tick_params(colors=self.TEXT_DIM, labelsize=8)
            ax.spines[:].set_color(self.BTN_BG)
            ax.grid(True, alpha=0.15, color=self.TEXT_DIM)

        # Exact vs linear
        axes[0].plot(data['deltas'], data['Z_exact'], color='#89b4fa',
                     linewidth=2.5, label=f'Exact:  Z = fB/(d₀+Δd)')
        axes[0].plot(data['deltas'], data['Z_lin'],   color='#f38ba8',
                     linewidth=2, linestyle='--',
                     label=f'Linear: Z₀ + ({data["slope"]:.4f})·Δd')
        axes[0].axvline(0, color=self.TEXT_DIM, linestyle=':', alpha=0.5)
        axes[0].axhline(data['Z0'], color=self.TEXT_DIM, linestyle=':', alpha=0.5)
        axes[0].set_xlabel('Δd (pixels)', color=self.TEXT, fontsize=9)
        axes[0].set_ylabel('Depth Z (m)', color=self.TEXT, fontsize=9)
        axes[0].set_title(f'Taylor Linearization at d₀={data["d0"]:.1f}px\n(§6.2)',
                           color=self.ACCENT2, fontsize=9)
        axes[0].legend(fontsize=8, facecolor=self.PANEL, labelcolor=self.TEXT)

        # Error of approximation
        lin_err = np.abs(data['Z_lin'] - data['Z_exact'])
        axes[1].plot(data['deltas'], lin_err * 100, color='#a6e3a1', linewidth=2.5)
        axes[1].fill_between(data['deltas'], 0, lin_err * 100, alpha=0.2, color='#a6e3a1')
        axes[1].set_xlabel('Δd (pixels)', color=self.TEXT, fontsize=9)
        axes[1].set_ylabel('Approximation error (cm)', color=self.TEXT, fontsize=9)
        axes[1].set_title('Linearization Error\n(valid for small Δd)', color=self.ACCENT2, fontsize=9)

        self.fig_3d.suptitle('Taylor Series Linearization of Z(d) = fB/d',
                              color=self.TEXT, fontsize=10, y=1.01)
        self.fig_3d.tight_layout()
        self.canvas_3d.draw()
        self.nb.select(5)
        self._log("Taylor linearization plot updated.", "success")

    def _show_histogram(self):
        if self.pipeline.depth_map is None:
            self._log("Compute depth map first.", "error")
            return

        self._log("\n── Depth Distribution ──", "accent")
        valid = self.pipeline.depth_map[~np.isnan(self.pipeline.depth_map)]
        valid = valid[(valid > 0) & (valid < 50)]

        self.fig_3d.clear()
        self.fig_3d.patch.set_facecolor('#12121e')
        axes = self.fig_3d.subplots(1, 2)

        for ax in axes:
            ax.set_facecolor('#12121e')
            ax.tick_params(colors=self.TEXT_DIM, labelsize=8)
            ax.spines[:].set_color(self.BTN_BG)
            ax.grid(True, alpha=0.15, color=self.TEXT_DIM)

        # Histogram
        n, bins, patches = axes[0].hist(valid, bins=60,
                                         color=self.ACCENT, edgecolor='none', alpha=0.85)
        med = np.median(valid)
        mea = np.mean(valid)
        axes[0].axvline(med, color='#f38ba8', linewidth=2, label=f'Median: {med:.3f}m')
        axes[0].axvline(mea, color='#f9e2af', linewidth=2, linestyle='--', label=f'Mean: {mea:.3f}m')
        axes[0].set_xlabel('Depth Z (m)', color=self.TEXT, fontsize=9)
        axes[0].set_ylabel('Pixel Count', color=self.TEXT, fontsize=9)
        axes[0].set_title('Depth Value Distribution', color=self.ACCENT2, fontsize=9)
        axes[0].legend(fontsize=8, facecolor=self.PANEL, labelcolor=self.TEXT)

        # Cumulative
        sorted_d = np.sort(valid)
        cdf = np.arange(1, len(sorted_d)+1) / len(sorted_d) * 100
        axes[1].plot(sorted_d, cdf, color='#a6e3a1', linewidth=2.5)
        axes[1].axhline(50, color=self.TEXT_DIM, linestyle=':', alpha=0.5, label='50th percentile')
        axes[1].axhline(90, color='#f9e2af',     linestyle=':', alpha=0.7, label='90th percentile')
        axes[1].set_xlabel('Depth Z (m)', color=self.TEXT, fontsize=9)
        axes[1].set_ylabel('Cumulative %', color=self.TEXT, fontsize=9)
        axes[1].set_title('Cumulative Depth Distribution', color=self.ACCENT2, fontsize=9)
        axes[1].legend(fontsize=8, facecolor=self.PANEL, labelcolor=self.TEXT)

        self.fig_3d.suptitle('Depth Map Statistics', color=self.TEXT, fontsize=10, y=1.01)
        self.fig_3d.tight_layout()
        self.canvas_3d.draw()
        self.nb.select(5)
        self._log(f"  Median: {med:.3f}m  |  Mean: {mea:.3f}m  |  Std: {np.std(valid):.3f}m", "success")

    def _save_results(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if not folder:
            return

        saved = []

        if self.pipeline.disparity is not None:
            d_img = self._disparity_as_colormap()
            if d_img is not None:
                cv2.imwrite(os.path.join(folder, 'disparity_map.png'), d_img)
                saved.append('disparity_map.png')

        if self.pipeline.depth_map is not None:
            d_img = self._depth_as_colormap()
            if d_img is not None:
                cv2.imwrite(os.path.join(folder, 'depth_map.png'), d_img)
                saved.append('depth_map.png')

            np.save(os.path.join(folder, 'depth_map.npy'), self.pipeline.depth_map)
            saved.append('depth_map.npy')

        # Save analysis figure
        try:
            self.fig_3d.savefig(os.path.join(folder, 'analysis_plots.png'),
                                 dpi=150, bbox_inches='tight',
                                 facecolor='#12121e')
            saved.append('analysis_plots.png')
        except Exception:
            pass

        # Save stats to text
        stats = self.pipeline.get_depth_stats()
        if stats:
            with open(os.path.join(folder, 'statistics.txt'), 'w') as f:
                f.write("Stereo Vision Depth Estimation Results\n")
                f.write("UBT 2026 — Mërgim Pirraku\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Focal length f  : {self.pipeline.focal_length:.2f} px\n")
                f.write(f"Baseline B      : {self.pipeline.baseline*100:.2f} cm\n")
                f.write(f"Sigma_d         : {self.pipeline.sigma_d:.2f} px\n\n")
                f.write("Depth Statistics:\n")
                for k, v in stats.items():
                    f.write(f"  {k:<14}: {v:.4f}\n")

                Z_med   = stats['median']
                sigma_z = (Z_med**2 / (self.pipeline.focal_length * self.pipeline.baseline)) * self.pipeline.sigma_d
                rel_err = sigma_z / Z_med * 100
                f.write(f"\nError at median depth ({Z_med:.3f}m):\n")
                f.write(f"  σ_Z = {sigma_z*100:.3f} cm\n")
                f.write(f"  σ_Z/Z = {rel_err:.3f}%\n")
            saved.append('statistics.txt')

        self._log(f"Saved {len(saved)} files to: {os.path.basename(folder)}", "success")
        for s in saved:
            self._log(f"  ✓ {s}", "dim")

        messagebox.showinfo("Saved", f"Saved {len(saved)} files to:\n{folder}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("Installing Pillow...")
        os.system("pip install Pillow")
        from PIL import Image, ImageTk

    app = StereoVisionApp()
    app.mainloop()