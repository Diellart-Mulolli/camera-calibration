
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
from pathlib import Path
from PIL import Image, ImageTk
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CORE ALGORITHMS (from paper)
# ============================================================

class StereoPipeline:
    """
    Implements all algorithms from the paper:
    - Zhang calibration  (Section 4.3)
    - Rectification      (Section 4.4)
    - Disparity          (Section 5.7)
    - Depth  Z = fB/d    (Section 5.8)
    - Error analysis     (Section 6)
    """

    def __init__(self):
        self.calib_data     = None
        self.rect_maps      = None
        self.img_left       = None
        self.img_right      = None
        self.img_left_rect  = None
        self.img_right_rect = None
        self.disparity      = None
        self.depth_map      = None
        self.points_3d      = None
        self._gt_disparity  = None

        # Default parameters (overridden from UI)
        self.focal_length = 1735.0   # pixels
        self.baseline     = 0.160    # metres
        self.num_disp     = 128
        self.block_size   = 15
        self.sigma_d      = 0.5      # disparity uncertainty (pixels)

    # ----------------------------------------------------------
    # CALIBRATION  §4.3
    # ----------------------------------------------------------
    def calibrate_stereo(self, left_path, right_path,
                         board_size=(9, 6), square_size=0.025,
                         log_fn=print):

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001)

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
        valid     = 0

        for lp, rp in zip(left_imgs, right_imgs):
            gl = cv2.cvtColor(cv2.imread(lp), cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2GRAY)
            img_shape = gl.shape[::-1]

            rl, cl = cv2.findChessboardCorners(gl, board_size, None)
            rr, cr = cv2.findChessboardCorners(gr, board_size, None)

            if rl and rr:
                objpoints.append(objp)
                imgpts_l.append(
                    cv2.cornerSubPix(gl, cl, (11, 11), (-1, -1), criteria))
                imgpts_r.append(
                    cv2.cornerSubPix(gr, cr, (11, 11), (-1, -1), criteria))
                valid += 1
                log_fn(f"  ✓ Pair {valid}: {os.path.basename(lp)}")
            else:
                log_fn(f"  ✗ Skipped: {os.path.basename(lp)}")

        if valid < 5:
            log_fn("ERROR: Need at least 5 valid pairs.")
            return False

        _, Kl, Dl, _, _ = cv2.calibrateCamera(
            objpoints, imgpts_l, img_shape, None, None)
        _, Kr, Dr, _, _ = cv2.calibrateCamera(
            objpoints, imgpts_r, img_shape, None, None)

        ret, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpts_l, imgpts_r,
            Kl, Dl, Kr, Dr, img_shape,
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=criteria
        )

        baseline = np.linalg.norm(T)

        self.calib_data = {
            'K_left':  Kl, 'D_left':  Dl,
            'K_right': Kr, 'D_right': Dr,
            'R': R, 'T': T, 'E': E, 'F': F,
            'baseline':   baseline,
            'image_size': img_shape,
            'rms':        ret
        }

        self.focal_length = float(Kl[0, 0])
        self.baseline     = float(baseline)

        log_fn(f"\n  Focal length : {self.focal_length:.2f} px")
        log_fn(f"  Baseline B   : {self.baseline * 100:.2f} cm")
        log_fn(f"  RMS error    : {ret:.4f} px")
        return True

    # ----------------------------------------------------------
    # RECTIFICATION  §4.4
    # ----------------------------------------------------------
    def compute_rectification(self, log_fn=print):
        if self.calib_data is None:
            log_fn("ERROR: Run calibration first.")
            return False

        c = self.calib_data
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            c['K_left'],  c['D_left'],
            c['K_right'], c['D_right'],
            c['image_size'], c['R'], c['T'],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )

        sz = c['image_size']
        mLx, mLy = cv2.initUndistortRectifyMap(
            c['K_left'],  c['D_left'],  R1, P1, sz, cv2.CV_32FC1)
        mRx, mRy = cv2.initUndistortRectifyMap(
            c['K_right'], c['D_right'], R2, P2, sz, cv2.CV_32FC1)

        self.rect_maps = {
            'map_left_x':  mLx, 'map_left_y':  mLy,
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
        self.img_left_rect  = cv2.remap(
            self.img_left,  m['map_left_x'],  m['map_left_y'],
            cv2.INTER_LANCZOS4)
        self.img_right_rect = cv2.remap(
            self.img_right, m['map_right_x'], m['map_right_y'],
            cv2.INTER_LANCZOS4)
        log_fn("  Rectification applied.")
        return True

    # ----------------------------------------------------------
    # DISPARITY  §5.7  d = xL − xR
    # ----------------------------------------------------------
    def compute_disparity(self, log_fn=print):
        left  = (self.img_left_rect
                 if self.img_left_rect  is not None else self.img_left)
        right = (self.img_right_rect
                 if self.img_right_rect is not None else self.img_right)

        if left is None:
            log_fn("ERROR: Load images first.")
            return False

        gl = (cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
              if len(left.shape)  == 3 else left)
        gr = (cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
              if len(right.shape) == 3 else right)

        nd = self.num_disp
        nd = nd if nd % 16 == 0 else (nd // 16 + 1) * 16
        bs = self.block_size
        bs = bs if bs % 2 == 1 else bs + 1
        bs = max(5, bs)

        stereo = cv2.StereoBM_create(numDisparities=nd, blockSize=bs)
        raw    = stereo.compute(gl, gr).astype(np.float32) / 16.0

        self.disparity = np.where(raw <= 0, np.nan, raw)

        valid_pct = (100 * np.sum(~np.isnan(self.disparity))
                     / self.disparity.size)
        log_fn(f"  Valid disparity pixels : {valid_pct:.1f}%")
        log_fn(f"  Disparity range        : "
               f"[{np.nanmin(self.disparity):.1f}, "
               f"{np.nanmax(self.disparity):.1f}] px")
        return True

    # ----------------------------------------------------------
    # DEPTH MAP  §5.8  Z = fB/d
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
            log_fn(f"  Depth range  : "
                   f"[{np.nanmin(self.depth_map):.3f}, "
                   f"{np.nanmax(self.depth_map):.3f}] m")
            log_fn(f"  Mean depth   : {np.nanmean(self.depth_map):.3f} m")
            log_fn(f"  Median depth : {np.nanmedian(self.depth_map):.3f} m")
        return True

    # ----------------------------------------------------------
    # ERROR ANALYSIS  §6
    # ----------------------------------------------------------
    def get_error_analysis_data(self, Z_range=(0.3, 10.0), n=300):
        f, B, s = self.focal_length, self.baseline, self.sigma_d
        Z    = np.linspace(Z_range[0], Z_range[1], n)
        d    = (f * B) / Z
        sZ   = (Z**2 / (f * B)) * s
        rZ   = sZ / Z * 100
        sens = (f * B) / (d**2)
        return {'Z': Z, 'd': d, 'sigma_Z': sZ,
                'rel_error': rZ, 'sensitivity': sens}

    def get_taylor_data(self, d0=None, delta_range=(-15, 15), n=300):
        if d0 is None:
            d0 = (np.nanmedian(self.disparity)
                  if self.disparity is not None else 30.0)
        f, B    = self.focal_length, self.baseline
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

    # ── Colour palette ───────────────────────────────────────
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
        self._dataset_entries = []

        self.title(
            "Stereo Vision Depth Estimation  |  UBT 2026  |  Mërgim Pirraku")
        self.geometry("1600x940")
        self.minsize(1200, 700)
        self.configure(bg=self.BG)
        self.resizable(True, True)

        self._build_styles()
        self._build_ui()

        self._log("Welcome! Load stereo images or calibration data to begin.",
                  "info")
        self._log(
            "Paper: Vlerësimi Gjeometrik i Thellësisë në Sistemet Stereo "
            "me Dy Kamera Statike", "dim")

    # ----------------------------------------------------------
    # STYLES
    # ----------------------------------------------------------
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use('clam')

        s.configure('TFrame',       background=self.BG)
        s.configure('Panel.TFrame', background=self.PANEL)
        s.configure('TLabel',
                    background=self.BG, foreground=self.TEXT,
                    font=('Segoe UI', 10))
        s.configure('TNotebook',     background=self.BG, borderwidth=0)
        s.configure('TNotebook.Tab',
                    background=self.BTN_BG, foreground=self.TEXT,
                    font=('Segoe UI', 10, 'bold'), padding=[14, 6])
        s.map('TNotebook.Tab',
              background=[('selected', self.ACCENT)],
              foreground=[('selected', '#ffffff')])
        s.configure('TScale',
                    background=self.PANEL, troughcolor=self.BTN_BG)
        s.configure('Horizontal.TProgressbar',
                    troughcolor=self.BTN_BG,
                    background=self.ACCENT,
                    thickness=6)
        s.configure('Vertical.TScrollbar',
                    background=self.BTN_BG,
                    troughcolor=self.PANEL,
                    arrowcolor=self.TEXT)

    def _make_btn(self, parent, text, command,
                  icon="", color=None, width=22):
        c = color or self.BTN_BG
        label = f"  {icon}  {text}" if icon else f"  {text}"
        b = tk.Button(
            parent, text=label, command=command,
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
                 font=('Segoe UI', 14, 'bold')
                 ).pack(side='left', padx=20, pady=10)
        tk.Label(top,
                 text="UBT 2026  |  Mërgim Pirraku",
                 bg=self.ACCENT, fg='#e0d7ff',
                 font=('Segoe UI', 10)
                 ).pack(side='right', padx=20)

        # ── Progress bar ─────────────────────────────────────
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(
            self, variable=self.progress_var,
            style='Horizontal.TProgressbar', maximum=100)
        self.progress.pack(fill='x', side='top')

        # ── Three-column body ─────────────────────────────────
        body = tk.Frame(self, bg=self.BG)
        body.pack(fill='both', expand=True)

        # Left column — fixed width, scrollable
        self._build_left_column(body)

        # Right column — fixed width, scrollable
        self._build_right_column(body)

        # Center column — fills remaining space
        center_col = tk.Frame(body, bg=self.BG)
        center_col.pack(side='left', fill='both', expand=True,
                        padx=4, pady=8)
        self._build_center_panel(center_col)

    # ── helper: scrollable side column ───────────────────────
    def _scrollable_col(self, parent, width, side):
        """Return an inner Frame that scrolls vertically."""
        outer = tk.Frame(parent, bg=self.BG, width=width)
        outer.pack(side=side, fill='y', padx=(8 if side == 'left' else 4,
                                               4 if side == 'left' else 8),
                   pady=8)
        outer.pack_propagate(False)

        canvas = tk.Canvas(outer, bg=self.BG, highlightthickness=0,
                           width=width - 16)
        sb = ttk.Scrollbar(outer, orient='vertical',
                           command=canvas.yview,
                           style='Vertical.TScrollbar')
        canvas.configure(yscrollcommand=sb.set)

        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        inner = tk.Frame(canvas, bg=self.BG)
        win_id = canvas.create_window((0, 0), window=inner, anchor='nw')

        def _on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox('all'))

        def _on_canvas_configure(e):
            canvas.itemconfig(win_id, width=e.width)

        inner.bind('<Configure>', _on_frame_configure)
        canvas.bind('<Configure>', _on_canvas_configure)

        # Mouse-wheel scrolling
        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')

        canvas.bind_all('<MouseWheel>', _on_mousewheel)
        return inner

    # ----------------------------------------------------------
    # LEFT PANEL  — controls
    # ----------------------------------------------------------
    def _build_left_column(self, parent):
        inner = self._scrollable_col(parent, width=295, side='left')
        self._build_left_panel(inner)

    def _build_left_panel(self, parent):

        def section(title, color=None):
            c = color or self.ACCENT
            f = tk.Frame(parent, bg=self.PANEL, bd=0)
            f.pack(fill='x', pady=(0, 6))
            tk.Label(f, text=f"  {title}",
                     bg=c, fg='white',
                     font=('Segoe UI', 10, 'bold'),
                     anchor='w').pack(fill='x', ipady=4)
            inner = tk.Frame(f, bg=self.PANEL)
            inner.pack(fill='x', padx=8, pady=6)
            return inner

        # ── 0. Dataset Browser ───────────────────────────────
        f0 = section("📦  0. Dataset Browser", color="#2d5a3d")
        self._build_dataset_browser_widgets(f0)

        # ── 1. Load Images ───────────────────────────────────
        f1 = section("📂  1. Load Images")
        self._make_btn(f1, "Load Left Image",
                       self._load_left,  "🖼", width=26
                       ).pack(fill='x', pady=2)
        self._make_btn(f1, "Load Right Image",
                       self._load_right, "🖼", width=26
                       ).pack(fill='x', pady=2)
        self._make_btn(f1, "Load Stereo Pair (folder)",
                       self._load_folder, "📁", width=26
                       ).pack(fill='x', pady=2)
        self.img_status = tk.Label(
            f1, text="No images loaded",
            bg=self.PANEL, fg=self.TEXT_DIM,
            font=('Segoe UI', 9), wraplength=240)
        self.img_status.pack(fill='x', pady=(4, 0))

        # ── 2. Calibration ───────────────────────────────────
        f2 = section("🔧  2. Calibration  (§4.3)")
        self._make_btn(f2, "Run Stereo Calibration",
                       self._run_calibration, "⚙", width=26
                       ).pack(fill='x', pady=2)
        self._make_btn(f2, "Load Saved Calibration",
                       self._load_calibration, "💾", width=26
                       ).pack(fill='x', pady=2)
        self._make_btn(f2, "Use Dataset Defaults",
                       self._use_defaults, "✔", width=26,
                       color="#2d5a3d"
                       ).pack(fill='x', pady=2)

        bf = tk.Frame(f2, bg=self.PANEL)
        bf.pack(fill='x', pady=(4, 0))
        tk.Label(bf, text="Board (cols × rows):",
                 bg=self.PANEL, fg=self.TEXT_DIM,
                 font=('Segoe UI', 9)).pack(anchor='w')
        brow = tk.Frame(bf, bg=self.PANEL)
        brow.pack(fill='x')
        self.board_cols = tk.IntVar(value=9)
        self.board_rows = tk.IntVar(value=6)
        tk.Spinbox(brow, from_=4, to=20,
                   textvariable=self.board_cols, width=4,
                   bg=self.BTN_BG, fg=self.TEXT, bd=0,
                   buttonbackground=self.BTN_BG
                   ).pack(side='left', padx=(0, 4))
        tk.Label(brow, text="×",
                 bg=self.PANEL, fg=self.TEXT).pack(side='left')
        tk.Spinbox(brow, from_=4, to=20,
                   textvariable=self.board_rows, width=4,
                   bg=self.BTN_BG, fg=self.TEXT, bd=0,
                   buttonbackground=self.BTN_BG
                   ).pack(side='left', padx=(4, 0))

        self.calib_status = tk.Label(
            f2, text="Not calibrated",
            bg=self.PANEL, fg=self.TEXT_DIM,
            font=('Segoe UI', 9), wraplength=240)
        self.calib_status.pack(fill='x', pady=(4, 0))

        # ── 3. Algorithm Params ──────────────────────────────
        f3 = section("⚙  3. Algorithm Parameters")

        def slider_row(par, label, var, from_, to, fmt="{:.0f}"):
            row = tk.Frame(par, bg=self.PANEL)
            row.pack(fill='x', pady=3)
            tk.Label(row, text=label,
                     bg=self.PANEL, fg=self.TEXT_DIM,
                     font=('Segoe UI', 9),
                     width=14, anchor='w').pack(side='left')
            val_lbl = tk.Label(row,
                               text=fmt.format(var.get()),
                               bg=self.PANEL, fg=self.ACCENT2,
                               font=('Segoe UI', 9, 'bold'), width=7)
            val_lbl.pack(side='right')

            def on_change(v, vl=val_lbl, f=fmt):
                vl.config(text=f.format(float(v)))

            ttk.Scale(row, from_=from_, to=to, variable=var,
                      orient='horizontal', command=on_change
                      ).pack(side='left', fill='x', expand=True, padx=4)

        self.num_disp_var = tk.IntVar(value=128)
        self.block_var    = tk.IntVar(value=15)
        self.sigma_d_var  = tk.DoubleVar(value=0.5)
        self.focal_var    = tk.DoubleVar(value=1735.0)
        self.baseline_var = tk.DoubleVar(value=16.0)

        slider_row(f3, "Num Disparities", self.num_disp_var, 16,  256)
        slider_row(f3, "Block Size",      self.block_var,     5,   51)
        slider_row(f3, "σ_d (pixels)",    self.sigma_d_var,  0.1, 3.0, "{:.2f}")
        slider_row(f3, "Focal f (px)",    self.focal_var,    200, 4000, "{:.0f}")
        slider_row(f3, "Baseline (cm)",   self.baseline_var,   1,   50, "{:.1f}")

        # ── 4. Pipeline ──────────────────────────────────────
        f4 = section("▶  4. Run Pipeline")
        self._make_btn(f4, "Rectify Images",
                       self._run_rectification, "↔", width=26,
                       color="#3a3a6a").pack(fill='x', pady=2)
        self._make_btn(f4, "Compute Disparity",
                       self._run_disparity, "📐", width=26,
                       color="#3a3a6a").pack(fill='x', pady=2)
        self._make_btn(f4, "Compute Depth Map",
                       self._run_depth, "📏", width=26,
                       color="#3a3a6a").pack(fill='x', pady=2)
        self._make_btn(f4, "▶▶  Run Full Pipeline",
                       self._run_full_pipeline, "🚀", width=26,
                       color=self.ACCENT).pack(fill='x', pady=(6, 2))

        # ── 5. Analysis & Export ─────────────────────────────
        f5 = section("📊  5. Analysis & Export")
        self._make_btn(f5, "Error Analysis Plots",
                       self._show_error_analysis, "📈", width=26
                       ).pack(fill='x', pady=2)
        self._make_btn(f5, "Taylor Linearization",
                       self._show_taylor, "∂", width=26
                       ).pack(fill='x', pady=2)
        self._make_btn(f5, "Depth Histogram",
                       self._show_histogram, "📊", width=26
                       ).pack(fill='x', pady=2)
        self._make_btn(f5, "Save Results",
                       self._save_results, "💾", width=26
                       ).pack(fill='x', pady=2)

    # ── Dataset browser widgets ───────────────────────────────
    def _build_dataset_browser_widgets(self, parent):
        btn_row = tk.Frame(parent, bg=self.PANEL)
        btn_row.pack(fill='x', pady=(0, 4))
        self._make_btn(btn_row, "Scan",
                       self._scan_datasets, "🔍",
                       color="#2d5a3d", width=10
                       ).pack(side='left', padx=(0, 4))
        self._make_btn(btn_row, "Download",
                       self._open_download_window, "⬇",
                       color="#3a2d5a", width=10
                       ).pack(side='left')

        list_frame = tk.Frame(parent, bg=self.PANEL)
        list_frame.pack(fill='x')
        sb = tk.Scrollbar(list_frame, bg=self.BTN_BG)
        sb.pack(side='right', fill='y')
        self.dataset_listbox = tk.Listbox(
            list_frame,
            bg=self.BTN_BG, fg=self.TEXT,
            font=('Consolas', 9),
            selectbackground=self.ACCENT,
            selectforeground='white',
            relief='flat', bd=0, height=5,
            yscrollcommand=sb.set)
        self.dataset_listbox.pack(fill='x')
        sb.config(command=self.dataset_listbox.yview)
        self.dataset_listbox.bind('<Double-Button-1>',
                                  self._load_selected_dataset)

        self.dataset_info = tk.Label(
            parent, text="Double-click to load",
            bg=self.PANEL, fg=self.TEXT_DIM,
            font=('Segoe UI', 8), wraplength=240)
        self.dataset_info.pack(fill='x', pady=(4, 0))

        self._make_btn(parent, "Load Selected",
                       self._load_selected_dataset, "▶",
                       color=self.ACCENT, width=26
                       ).pack(fill='x', pady=(4, 0))

        self.after(600, self._scan_datasets)

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

        self.canvases    = {}
        self.tab_frames  = {}

        for name, init_fn in tabs:
            frame = ttk.Frame(self.nb, style='TFrame')
            self.nb.add(frame, text=f"  {name}  ")
            self.tab_frames[name] = frame
            init_fn(frame)

    def _make_image_canvas(self, parent, name):
        canvas = tk.Canvas(parent, bg='#12121e', highlightthickness=0)
        canvas.pack(fill='both', expand=True)
        self.canvases[name] = canvas
        canvas.bind('<Configure>', lambda e, n=name: self._resize_canvas(n))
        return canvas

    def _tab_left(self, f):
        self._make_image_canvas(f, 'left')

    def _tab_right(self, f):
        self._make_image_canvas(f, 'right')

    def _tab_rect(self, f):
        self._make_image_canvas(f, 'rect')

    def _tab_disp(self, f):
        self._make_image_canvas(f, 'disp')

    def _tab_depth(self, f):
        self._make_image_canvas(f, 'depth')

    def _tab_3d(self, f):
        self.fig_3d = Figure(figsize=(8, 5), dpi=96,
                             facecolor='#12121e')
        self.ax_3d  = self.fig_3d.add_subplot(111)
        self.ax_3d.set_facecolor('#12121e')
        self.ax_3d.text(0.5, 0.5, 'Run analysis to see plots',
                        ha='center', va='center',
                        color=self.TEXT_DIM, fontsize=13,
                        transform=self.ax_3d.transAxes)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=f)
        self.canvas_3d.get_tk_widget().pack(fill='both', expand=True)

    # ----------------------------------------------------------
    # RIGHT PANEL  — statistics + log   (FIX: scrollable)
    # ----------------------------------------------------------
    def _build_right_column(self, parent):
        inner = self._scrollable_col(parent, width=320, side='right')
        self._build_right_panel(inner)

    def _build_right_panel(self, parent):

        def section(title):
            f = tk.Frame(parent, bg=self.PANEL, bd=0)
            f.pack(fill='x', pady=(0, 6))
            tk.Label(f, text=f"  {title}",
                     bg=self.ACCENT, fg='white',
                     font=('Segoe UI', 10, 'bold'),
                     anchor='w').pack(fill='x', ipady=4)
            inner = tk.Frame(f, bg=self.PANEL)
            inner.pack(fill='x', padx=10, pady=8)
            return inner

        # ── Statistics ───────────────────────────────────────
        s = section("📊  Statistics")
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
            row = tk.Frame(s, bg=self.PANEL)
            row.pack(fill='x', pady=2)
            tk.Label(row, text=label + ":",
                     bg=self.PANEL, fg=self.TEXT_DIM,
                     font=('Segoe UI', 9),
                     anchor='w', width=18).pack(side='left')
            var = tk.StringVar(value="—")
            self._stat_vars[key] = var
            tk.Label(row, textvariable=var,
                     bg=self.PANEL, fg=self.ACCENT2,
                     font=('Segoe UI', 10, 'bold'),
                     anchor='e').pack(side='left', padx=(4, 0))
            tk.Label(row, text=unit,
                     bg=self.PANEL, fg=self.TEXT_DIM,
                     font=('Segoe UI', 9)).pack(side='left', padx=(2, 0))

        ttk.Separator(parent).pack(fill='x', pady=4)

        # ── Key Formulas ─────────────────────────────────────
        rf = section("📐  Key Formulas")
        formulas = [
            ("Depth (§5.8)",       "Z = f·B / d"),
            ("Disparity (§5.7)",   "d = xL − xR"),
            ("Error (§6.3)",       "σ_Z = (Z²/fB)·σ_d"),
            ("Sensitivity (§6.2)", "dZ/dd = −fB/d²"),
            ("Rel. err (§6.3)",    "σ_Z/Z = (Z/fB)·σ_d"),
        ]
        for name, formula in formulas:
            row = tk.Frame(rf, bg=self.PANEL)
            row.pack(fill='x', pady=2)
            tk.Label(row, text=name,
                     bg=self.PANEL, fg=self.TEXT_DIM,
                     font=('Segoe UI', 8),
                     width=18, anchor='w').pack(side='left')
            tk.Label(row, text=formula,
                     bg=self.PANEL, fg=self.INFO,
                     font=('Consolas', 9, 'bold')).pack(side='left')

        ttk.Separator(parent).pack(fill='x', pady=4)

        # ── Console Log ──────────────────────────────────────
        log_outer = tk.Frame(parent, bg=self.PANEL)
        log_outer.pack(fill='both', expand=True)

        tk.Label(log_outer, text="  📋  Console Log",
                 bg=self.ACCENT, fg='white',
                 font=('Segoe UI', 10, 'bold'),
                 anchor='w').pack(fill='x', ipady=4)

        log_body = tk.Frame(log_outer, bg=self.PANEL)
        log_body.pack(fill='both', expand=True)

        self.log_text = tk.Text(
            log_body,
            bg='#12121e', fg=self.TEXT,
            font=('Consolas', 9), relief='flat', bd=0,
            wrap='word', state='disabled',
            width=36,      # ← fixed character width stops overflow
            height=22)
        sb_log = ttk.Scrollbar(log_body, command=self.log_text.yview,
                               style='Vertical.TScrollbar')
        self.log_text.configure(yscrollcommand=sb_log.set)
        sb_log.pack(side='right', fill='y')
        self.log_text.pack(side='left', fill='both', expand=True,
                           padx=2, pady=2)

        for tag, color in [
            ('info',    self.INFO),
            ('success', self.SUCCESS),
            ('warning', self.WARNING),
            ('error',   self.ERROR),
            ('dim',     self.TEXT_DIM),
            ('accent',  self.ACCENT2),
        ]:
            self.log_text.tag_config(tag, foreground=color)

        self._make_btn(log_outer, "Clear Log",
                       self._clear_log, width=14
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
        if img_np is None:
            return None
        h, w   = img_np.shape[:2]
        scale  = min(target_w / w, target_h / h) * 0.97
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
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
        canvas._photo = photo   # prevent GC

    def _resize_canvas(self, name):
        img_map = {
            'left':  self.pipeline.img_left,
            'right': self.pipeline.img_right,
            'disp':  self._disparity_as_colormap(),
            'depth': self._depth_as_colormap(),
        }
        if name == 'rect' and self.pipeline.img_left_rect is not None:
            self._display_on_canvas('rect', self._rectified_combined())
        elif name in img_map and img_map[name] is not None:
            self._display_on_canvas(name, img_map[name])

    # ----------------------------------------------------------
    # COLOURMAP HELPERS
    # ----------------------------------------------------------
    def _disparity_as_colormap(self):
        if self.pipeline.disparity is None:
            return None
        d = np.nan_to_num(self.pipeline.disparity, nan=0
                          ).astype(np.float32)
        d_norm = cv2.normalize(d, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)

    def _depth_as_colormap(self):
        if self.pipeline.depth_map is None:
            return None
        d  = np.nan_to_num(self.pipeline.depth_map, nan=0
                           ).astype(np.float32)
        valid = d[d > 0]
        p95   = np.percentile(valid, 95) if len(valid) else 1.0
        d     = np.clip(d, 0, p95)
        d_norm = cv2.normalize(d, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)

    # ----------------------------------------------------------
    # RECTIFIED VIEW WITH CORRECT EPIPOLAR LINES  (FIX)
    # ----------------------------------------------------------
    def _rectified_combined(self):
        """
        Side-by-side rectified pair with TRUE horizontal epipolar lines.

        After stereo rectification every epipolar line is a horizontal
        scan-line shared by both images.  We draw N evenly-spaced
        horizontal lines that span the full width of the composite image.
        A short vertical tick at the centre of each half confirms
        alignment.
        """
        if self.pipeline.img_left_rect is None:
            return None

        L = self.pipeline.img_left_rect.copy()
        R = self.pipeline.img_right_rect.copy()

        # Ensure both are colour BGR
        if len(L.shape) == 2:
            L = cv2.cvtColor(L, cv2.COLOR_GRAY2BGR)
        if len(R.shape) == 2:
            R = cv2.cvtColor(R, cv2.COLOR_GRAY2BGR)

        # Match heights (crop bottom if necessary)
        h = min(L.shape[0], R.shape[0])
        L, R = L[:h], R[:h]

        combined = np.hstack([L, R])
        total_w  = combined.shape[1]
        half_w   = L.shape[1]

        # ── Draw horizontal epipolar lines ───────────────────
        num_lines = 12
        line_color  = (0, 200, 80)    # green
        tick_color  = (255, 180,  0)  # amber ticks
        line_thick  = 1
        tick_len    = 14

        for i in range(1, num_lines + 1):
            y = int(i * h / (num_lines + 1))

            # Full-width horizontal line
            cv2.line(combined, (0, y), (total_w - 1, y),
                     line_color, line_thick, cv2.LINE_AA)

            # Short vertical tick on left half centre
            cx_l = half_w // 2
            cv2.line(combined,
                     (cx_l, y - tick_len // 2),
                     (cx_l, y + tick_len // 2),
                     tick_color, 2, cv2.LINE_AA)

            # Short vertical tick on right half centre
            cx_r = half_w + half_w // 2
            cv2.line(combined,
                     (cx_r, y - tick_len // 2),
                     (cx_r, y + tick_len // 2),
                     tick_color, 2, cv2.LINE_AA)

        # ── Dividing line between the two images ─────────────
        cv2.line(combined, (half_w, 0), (half_w, h - 1),
                 (150, 150, 255), 2)

        return combined

    # ----------------------------------------------------------
    # STATISTICS UPDATE
    # ----------------------------------------------------------
    def _update_stats(self):
        p     = self.pipeline
        stats = p.get_depth_stats()

        self._stat_vars['focal'].set(f"{p.focal_length:.1f}")
        self._stat_vars['baseline'].set(f"{p.baseline * 100:.2f}")

        if stats:
            for k in ('min', 'max', 'mean', 'median', 'std'):
                self._stat_vars[k].set(f"{stats[k]:.3f}")
            self._stat_vars['valid'].set(f"{stats['valid']:.1f}")

            Z_med   = stats['median']
            sigma_z = (Z_med**2 / (p.focal_length * p.baseline)) * p.sigma_d
            rel_err = (Z_med / (p.focal_length * p.baseline)) * p.sigma_d * 100
            self._stat_vars['sigma_z'].set(f"{sigma_z * 100:.3f}")
            self._stat_vars['rel_err'].set(f"{rel_err:.3f}")

    # ----------------------------------------------------------
    # BUTTON HANDLERS
    # ----------------------------------------------------------
    def _load_left(self):
        path = filedialog.askopenfilename(
            title="Select Left Image",
            filetypes=[("Images", "*.png *.jpg *.bmp *.tiff"),
                       ("All", "*.*")])
        if path:
            self.pipeline.img_left = cv2.imread(path)
            self._log(f"Left: {os.path.basename(path)}", "success")
            self.img_status.config(
                text=f"L: {os.path.basename(path)}", fg=self.SUCCESS)
            self.after(100, lambda: self._display_on_canvas(
                'left', self.pipeline.img_left))
            self.nb.select(0)

    def _load_right(self):
        path = filedialog.askopenfilename(
            title="Select Right Image",
            filetypes=[("Images", "*.png *.jpg *.bmp *.tiff"),
                       ("All", "*.*")])
        if path:
            self.pipeline.img_right = cv2.imread(path)
            self._log(f"Right: {os.path.basename(path)}", "success")
            self.img_status.config(
                text=f"R: {os.path.basename(path)}", fg=self.SUCCESS)
            self.after(100, lambda: self._display_on_canvas(
                'right', self.pipeline.img_right))
            self.nb.select(1)

    def _load_folder(self):
        folder = filedialog.askdirectory(
            title="Select folder with left.png & right.png")
        if folder:
            candidates = [
                ('left.png',  'right.png'),
                ('im0.png',   'im1.png'),
                ('left.jpg',  'right.jpg'),
            ]
            lp = rp = None
            for lname, rname in candidates:
                _l = os.path.join(folder, lname)
                _r = os.path.join(folder, rname)
                if os.path.exists(_l) and os.path.exists(_r):
                    lp, rp = _l, _r
                    break

            if lp:
                self.pipeline.img_left  = cv2.imread(lp)
                self.pipeline.img_right = cv2.imread(rp)
                h, w = self.pipeline.img_left.shape[:2]
                self._log(
                    f"Loaded pair from: {os.path.basename(folder)}"
                    f"  ({w}×{h})", "success")
                self.img_status.config(
                    text=f"Pair: {os.path.basename(folder)}",
                    fg=self.SUCCESS)
                self.after(100, lambda: self._display_on_canvas(
                    'left',  self.pipeline.img_left))
                self.after(150, lambda: self._display_on_canvas(
                    'right', self.pipeline.img_right))
            else:
                self._log("No left/right pair found in folder.", "error")

    def _run_calibration(self):
        ldir = filedialog.askdirectory(
            title="LEFT calibration images folder")
        if not ldir:
            return
        rdir = filedialog.askdirectory(
            title="RIGHT calibration images folder")
        if not rdir:
            return

        def task():
            self._log("\n── Stereo Calibration (§4.3) ──", "accent")
            self._set_progress(10)
            ok = self.pipeline.calibrate_stereo(
                ldir, rdir,
                board_size=(self.board_cols.get(),
                            self.board_rows.get()),
                log_fn=self._log)
            self._set_progress(60)
            if ok:
                ok2 = self.pipeline.compute_rectification(self._log)
                self._set_progress(90)
                if ok2:
                    self.calib_status.config(
                        text=(f"✓ RMS="
                              f"{self.pipeline.calib_data['rms']:.3f} px"),
                        fg=self.SUCCESS)
                    self._log("Calibration complete!", "success")
                    self._update_stats()
            else:
                self.calib_status.config(
                    text="✗ Failed", fg=self.ERROR)
            self._set_progress(100)
            self.after(1500, lambda: self._set_progress(0))

        threading.Thread(target=task, daemon=True).start()

    def _load_calibration(self):
        path = filedialog.askopenfilename(
            title="Load calibration .npz",
            filetypes=[("NumPy archive", "*.npz"), ("All", "*.*")])
        if path:
            try:
                data = np.load(path)
                self.pipeline.calib_data = {
                    k: data[k] for k in data.files}
                self.pipeline.baseline     = float(data['baseline'])
                self.pipeline.focal_length = float(data['K_left'][0, 0])
                self.pipeline.compute_rectification(self._log)
                self.calib_status.config(
                    text=f"✓ {os.path.basename(path)}", fg=self.SUCCESS)
                self._log(f"Loaded: {os.path.basename(path)}", "success")
                self._update_stats()
            except Exception as e:
                self._log(f"Load failed: {e}", "error")

    def _use_defaults(self):
        self.pipeline.focal_length = 1735.0
        self.pipeline.baseline     = 0.160
        self.focal_var.set(1735.0)
        self.baseline_var.set(16.0)
        self.calib_status.config(
            text="✓ Middlebury defaults", fg=self.WARNING)
        self._log("Using defaults: f=1735 px, B=16 cm", "warning")
        self._update_stats()

    def _sync_params(self):
        nd = int(self.num_disp_var.get())
        nd = nd if nd % 16 == 0 else (nd // 16) * 16
        self.pipeline.num_disp     = max(16, nd)
        bs = int(self.block_var.get())
        self.pipeline.block_size   = bs + 1 if bs % 2 == 0 else bs
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
                self.pipeline.img_left_rect  = self.pipeline.img_left
                self.pipeline.img_right_rect = self.pipeline.img_right
                ok = True
                self._log("  Pre-rectified (no calib data).", "warning")
            self._set_progress(80)
            if ok:
                combined = self._rectified_combined()
                self.after(100, lambda: self._display_on_canvas(
                    'rect', combined))
                self.nb.select(2)
                self._log(
                    "Done — green lines are horizontal epipolar lines.",
                    "success")
            self._set_progress(100)
            self.after(1500, lambda: self._set_progress(0))

        threading.Thread(target=task, daemon=True).start()

    def _run_disparity(self):
        self._sync_params()

        def task():
            self._log("\n── Disparity  d = xL−xR  (§5.7) ──", "accent")
            self._set_progress(15)
            ok = self.pipeline.compute_disparity(self._log)
            self._set_progress(85)
            if ok:
                img = self._disparity_as_colormap()
                self.after(100, lambda: self._display_on_canvas('disp', img))
                self.nb.select(3)
                self._log("Done (plasma: bright=near).", "success")
            self._set_progress(100)
            self.after(1500, lambda: self._set_progress(0))

        threading.Thread(target=task, daemon=True).start()

    def _run_depth(self):
        self._sync_params()

        def task():
            self._log("\n── Depth  Z = f·B/d  (§5.8) ──", "accent")
            self._set_progress(15)
            ok = self.pipeline.compute_depth(self._log)
            self._set_progress(85)
            if ok:
                img = self._depth_as_colormap()
                self.after(100, lambda: self._display_on_canvas('depth', img))
                self.nb.select(4)
                self._update_stats()
                self._log("Done (inferno: bright=far).", "success")
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

            self._log("1/3 — Rectification…", "info")
            self._set_progress(15)
            if self.pipeline.calib_data:
                self.pipeline.apply_rectification(self._log)
            else:
                self.pipeline.img_left_rect  = self.pipeline.img_left
                self.pipeline.img_right_rect = self.pipeline.img_right
                self._log("  Pre-rectified.", "warning")
            self.after(100, lambda: self._display_on_canvas(
                'rect', self._rectified_combined()))
            self._set_progress(33)

            self._log("2/3 — Disparity…", "info")
            self.pipeline.compute_disparity(self._log)
            self.after(100, lambda: self._display_on_canvas(
                'disp', self._disparity_as_colormap()))
            self._set_progress(66)

            self._log("3/3 — Depth…", "info")
            self.pipeline.compute_depth(self._log)
            self.after(100, lambda: self._display_on_canvas(
                'depth', self._depth_as_colormap()))
            self._set_progress(90)

            self._update_stats()
            self._log("\n✓ Pipeline complete!", "success")
            self.nb.select(4)
            self._set_progress(100)
            self.after(1500, lambda: self._set_progress(0))

        threading.Thread(target=task, daemon=True).start()

    # ----------------------------------------------------------
    # ANALYSIS PLOTS
    # ----------------------------------------------------------
    def _show_error_analysis(self):
        self._sync_params()
        p    = self.pipeline
        data = p.get_error_analysis_data()
        Z, sZ, rZ = data['Z'], data['sigma_Z'], data['rel_error']

        self.fig_3d.clear()
        self.fig_3d.patch.set_facecolor('#12121e')
        axes = self.fig_3d.subplots(1, 3)
        bg   = '#12121e'

        def style(ax, xlabel, ylabel, title):
            ax.set_facecolor(bg)
            ax.set_xlabel(xlabel, color=self.TEXT,    fontsize=9)
            ax.set_ylabel(ylabel, color=self.TEXT,    fontsize=9)
            ax.set_title(title,   color=self.ACCENT2, fontsize=9)
            ax.tick_params(colors=self.TEXT_DIM, labelsize=8)
            ax.spines[:].set_color(self.BTN_BG)
            ax.grid(True, alpha=0.15, color=self.TEXT_DIM)

        axes[0].plot(Z, sZ * 100, color='#f38ba8', linewidth=2.5)
        axes[0].fill_between(Z, 0, sZ * 100, alpha=0.2, color='#f38ba8')
        style(axes[0], 'Z (m)', 'σ_Z (cm)',
              'Absolute Error\nσ_Z = Z²/(fB)·σ_d')

        axes[1].plot(Z, rZ, color='#89b4fa', linewidth=2.5)
        axes[1].fill_between(Z, 0, rZ, alpha=0.2, color='#89b4fa')
        style(axes[1], 'Z (m)', 'σ_Z/Z (%)',
              'Relative Error\nσ_Z/Z = (Z/fB)·σ_d')

        for b, c in zip([0.05, 0.10, 0.20, 0.30],
                        ['#f38ba8', '#f9e2af', '#a6e3a1', '#89b4fa']):
            sZ_b = (Z**2 / (p.focal_length * b)) * p.sigma_d
            axes[2].plot(Z, sZ_b * 100, color=c, linewidth=2,
                         label=f'B={b*100:.0f}cm')
        style(axes[2], 'Z (m)', 'σ_Z (cm)',
              'Effect of Baseline\n(larger B → more accurate)')
        axes[2].legend(fontsize=8,
                       facecolor=self.PANEL, labelcolor=self.TEXT)

        self.fig_3d.suptitle(
            f'Error Analysis  '
            f'(f={p.focal_length:.0f}px, '
            f'B={p.baseline*100:.1f}cm, '
            f'σ_d={p.sigma_d:.2f}px)',
            color=self.TEXT, fontsize=10, y=1.01)
        self.fig_3d.tight_layout()
        self.canvas_3d.draw()
        self.nb.select(5)
        self._log("Error analysis updated.", "success")

    def _show_taylor(self):
        self._sync_params()
        p    = self.pipeline
        data = p.get_taylor_data()

        self.fig_3d.clear()
        self.fig_3d.patch.set_facecolor('#12121e')
        axes = self.fig_3d.subplots(1, 2)

        for ax in axes:
            ax.set_facecolor('#12121e')
            ax.tick_params(colors=self.TEXT_DIM, labelsize=8)
            ax.spines[:].set_color(self.BTN_BG)
            ax.grid(True, alpha=0.15, color=self.TEXT_DIM)

        axes[0].plot(data['deltas'], data['Z_exact'],
                     color='#89b4fa', linewidth=2.5,
                     label='Exact Z = fB/(d₀+Δd)')
        axes[0].plot(data['deltas'], data['Z_lin'],
                     color='#f38ba8', linewidth=2, linestyle='--',
                     label=f'Linear  (slope={data["slope"]:.4f})')
        axes[0].axvline(0, color=self.TEXT_DIM, linestyle=':', alpha=0.5)
        axes[0].axhline(data['Z0'], color=self.TEXT_DIM,
                        linestyle=':', alpha=0.5)
        axes[0].set_xlabel('Δd (px)', color=self.TEXT, fontsize=9)
        axes[0].set_ylabel('Z (m)',   color=self.TEXT, fontsize=9)
        axes[0].set_title(
            f'Taylor at d₀={data["d0"]:.1f}px  (§6.2)',
            color=self.ACCENT2, fontsize=9)
        axes[0].legend(fontsize=8,
                       facecolor=self.PANEL, labelcolor=self.TEXT)

        err = np.abs(data['Z_lin'] - data['Z_exact'])
        axes[1].plot(data['deltas'], err * 100,
                     color='#a6e3a1', linewidth=2.5)
        axes[1].fill_between(data['deltas'], 0, err * 100,
                             alpha=0.2, color='#a6e3a1')
        axes[1].set_xlabel('Δd (px)',              color=self.TEXT, fontsize=9)
        axes[1].set_ylabel('Approx. error (cm)',   color=self.TEXT, fontsize=9)
        axes[1].set_title('Linearisation Error\n(valid for small Δd)',
                           color=self.ACCENT2, fontsize=9)

        self.fig_3d.suptitle('Taylor Series Linearisation of Z(d) = fB/d',
                              color=self.TEXT, fontsize=10, y=1.01)
        self.fig_3d.tight_layout()
        self.canvas_3d.draw()
        self.nb.select(5)
        self._log("Taylor plot updated.", "success")

    def _show_histogram(self):
        if self.pipeline.depth_map is None:
            self._log("Compute depth map first.", "error")
            return

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

        med = np.median(valid)
        mea = np.mean(valid)
        axes[0].hist(valid, bins=60,
                     color=self.ACCENT, edgecolor='none', alpha=0.85)
        axes[0].axvline(med, color='#f38ba8', linewidth=2,
                        label=f'Median: {med:.3f}m')
        axes[0].axvline(mea, color='#f9e2af', linewidth=2,
                        linestyle='--', label=f'Mean: {mea:.3f}m')
        axes[0].set_xlabel('Z (m)',        color=self.TEXT, fontsize=9)
        axes[0].set_ylabel('Pixel count',  color=self.TEXT, fontsize=9)
        axes[0].set_title('Depth Distribution',
                           color=self.ACCENT2, fontsize=9)
        axes[0].legend(fontsize=8,
                       facecolor=self.PANEL, labelcolor=self.TEXT)

        sorted_d = np.sort(valid)
        cdf      = np.arange(1, len(sorted_d) + 1) / len(sorted_d) * 100
        axes[1].plot(sorted_d, cdf, color='#a6e3a1', linewidth=2.5)
        axes[1].axhline(50, color=self.TEXT_DIM, linestyle=':',
                        alpha=0.5, label='50th pct')
        axes[1].axhline(90, color='#f9e2af', linestyle=':',
                        alpha=0.7, label='90th pct')
        axes[1].set_xlabel('Z (m)',      color=self.TEXT, fontsize=9)
        axes[1].set_ylabel('CDF (%)',    color=self.TEXT, fontsize=9)
        axes[1].set_title('Cumulative Distribution',
                           color=self.ACCENT2, fontsize=9)
        axes[1].legend(fontsize=8,
                       facecolor=self.PANEL, labelcolor=self.TEXT)

        self.fig_3d.suptitle('Depth Map Statistics',
                              color=self.TEXT, fontsize=10, y=1.01)
        self.fig_3d.tight_layout()
        self.canvas_3d.draw()
        self.nb.select(5)
        self._log(
            f"  Median={med:.3f}m  Mean={mea:.3f}m  "
            f"Std={np.std(valid):.3f}m", "success")

    def _save_results(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if not folder:
            return
        saved = []

        if self.pipeline.disparity is not None:
            img = self._disparity_as_colormap()
            if img is not None:
                cv2.imwrite(os.path.join(folder, 'disparity_map.png'), img)
                saved.append('disparity_map.png')

        if self.pipeline.depth_map is not None:
            img = self._depth_as_colormap()
            if img is not None:
                cv2.imwrite(os.path.join(folder, 'depth_map.png'), img)
                saved.append('depth_map.png')
            np.save(os.path.join(folder, 'depth_map.npy'),
                    self.pipeline.depth_map)
            saved.append('depth_map.npy')

        try:
            self.fig_3d.savefig(
                os.path.join(folder, 'analysis_plots.png'),
                dpi=150, bbox_inches='tight', facecolor='#12121e')
            saved.append('analysis_plots.png')
        except Exception:
            pass

        stats = self.pipeline.get_depth_stats()
        if stats:
            with open(os.path.join(folder, 'statistics.txt'), 'w') as fh:
                fh.write("Stereo Vision Depth Estimation\n")
                fh.write("UBT 2026 — Mërgim Pirraku\n")
                fh.write("=" * 40 + "\n\n")
                fh.write(f"f  = {self.pipeline.focal_length:.2f} px\n")
                fh.write(f"B  = {self.pipeline.baseline*100:.2f} cm\n")
                fh.write(f"σ_d= {self.pipeline.sigma_d:.2f} px\n\n")
                for k, v in stats.items():
                    fh.write(f"  {k:<14}: {v:.4f}\n")
                Z_med   = stats['median']
                sigma_z = (Z_med**2
                           / (self.pipeline.focal_length
                              * self.pipeline.baseline)
                           ) * self.pipeline.sigma_d
                fh.write(f"\nσ_Z @ median = {sigma_z*100:.3f} cm\n")
                fh.write(f"σ_Z/Z        = {sigma_z/Z_med*100:.3f} %\n")
            saved.append('statistics.txt')

        self._log(
            f"Saved {len(saved)} file(s) to {os.path.basename(folder)}",
            "success")
        messagebox.showinfo("Saved",
                            f"Saved {len(saved)} files to:\n{folder}")

    # ----------------------------------------------------------
    # DATASET BROWSER METHODS
    # ----------------------------------------------------------
    def _scan_datasets(self):
        """Scan datasets/ folder."""
        datasets_dir = Path(__file__).parent / "datasets"
        if not datasets_dir.exists():
            self.dataset_listbox.delete(0, 'end')
            self.dataset_listbox.insert('end', "  datasets/ not found")
            return

        # Simple scan: look for folders containing im0.png / left.png
        entries = []
        for d in sorted(datasets_dir.iterdir()):
            if not d.is_dir():
                continue
            left  = (d / 'im0.png'   if (d / 'im0.png').exists()
                     else d / 'left.png')
            right = (d / 'im1.png'   if (d / 'im1.png').exists()
                     else d / 'right.png')
            if left.exists() and right.exists():
                has_gt = (d / 'disp0GT.pfm').exists()
                entries.append({'name': d.name,
                                'left': left, 'right': right,
                                'has_gt': has_gt,
                                'meta': {}})

        self._dataset_entries = entries
        self.dataset_listbox.delete(0, 'end')

        if not entries:
            self.dataset_listbox.insert('end', "  No datasets found")
            self.dataset_listbox.insert('end', "  Click Download →")
            self.dataset_info.config(
                text="Use Download to get datasets",
                fg=self.WARNING)
            return

        for e in entries:
            gt = " [GT]" if e['has_gt'] else ""
            self.dataset_listbox.insert('end', f"  {e['name']}{gt}")

        self.dataset_info.config(
            text=f"{len(entries)} dataset(s) found. Double-click to load.",
            fg=self.SUCCESS)
        self._log(f"Datasets found: {len(entries)}", "success")

    def _load_selected_dataset(self, event=None):
        sel = self.dataset_listbox.curselection()
        if not sel:
            return
        entry = self._dataset_entries[sel[0]]
        left  = cv2.imread(str(entry['left']))
        right = cv2.imread(str(entry['right']))
        if left is None or right is None:
            self._log("Could not read dataset images.", "error")
            return

        self.pipeline.img_left        = left
        self.pipeline.img_right       = right
        self.pipeline.img_left_rect   = left
        self.pipeline.img_right_rect  = right

        h, w = left.shape[:2]
        self._log(f"Loaded: {entry['name']}  {w}×{h}", "success")
        self.img_status.config(
            text=f"Dataset: {entry['name']}", fg=self.SUCCESS)
        self.dataset_info.config(
            text=f"Loaded: {entry['name']} ({w}×{h})", fg=self.SUCCESS)

        self.after(100, lambda: self._display_on_canvas('left',  left))
        self.after(150, lambda: self._display_on_canvas('right', right))
        self.nb.select(0)

    def _open_download_window(self):
        win = tk.Toplevel(self)
        win.title("Dataset Download Manager")
        win.geometry("560x260")
        win.configure(bg=self.BG)
        tk.Label(win,
                 text="  ⬇  Dataset Download Manager",
                 bg=self.ACCENT, fg='white',
                 font=('Segoe UI', 12, 'bold'),
                 anchor='w').pack(fill='x', ipady=8)
        tk.Label(win,
                 text=(
                     "Download Middlebury datasets manually from:\n"
                     "https://vision.middlebury.edu/stereo/data/\n\n"
                     "Place folders inside  datasets/  next to this script.\n"
                     "Each folder should contain  im0.png  and  im1.png."),
                 bg=self.BG, fg=self.TEXT,
                 font=('Segoe UI', 10), justify='left',
                 padx=20, pady=10).pack(fill='both', expand=True)
        self._make_btn(win, "Open datasets/ folder",
                       lambda: os.startfile(
                           str(Path(__file__).parent / "datasets")),
                       "📂", color=self.ACCENT, width=28
                       ).pack(pady=10)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app = StereoVisionApp()
    app.mainloop()