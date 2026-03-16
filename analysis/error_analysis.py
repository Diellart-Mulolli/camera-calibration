# error_analysis.py
# Error Analysis — Paper Section 6
# σ_Z = (Z² / f·B) · σ_d
# Student: Mërgim Pirraku | UBT 2026

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


# ── Color palette matching the GUI ──────────────────────────
COLORS = {
    'bg':       '#12121e',
    'panel':    '#1e1e2e',
    'accent':   '#7c6af7',
    'accent2':  '#56cfb2',
    'text':     '#cdd6f4',
    'dimtext':  '#6c7086',
    'red':      '#f38ba8',
    'yellow':   '#f9e2af',
    'green':    '#a6e3a1',
    'blue':     '#89b4fa',
}


def absolute_depth_error(Z, f, B, sigma_d):
    """σ_Z = (Z² / f·B) · σ_d  — Eq. from Section 6.3"""
    return (Z**2 / (f * B)) * sigma_d


def relative_depth_error(Z, f, B, sigma_d):
    """σ_Z/Z = (Z / f·B) · σ_d  — Section 6.3"""
    return (Z / (f * B)) * sigma_d * 100  # percent


def depth_sensitivity(d, f, B):
    """dZ/dd = -f·B / d²  — Section 6.2"""
    return -(f * B) / (d**2)


def taylor_linearization(d0, f, B, delta_range=(-15, 15), n=400):
    """
    First-order Taylor expansion of Z(d) = fB/d around d0.
    Z ≈ Z0 + (dZ/dd)|d0 · Δd
    """
    deltas  = np.linspace(delta_range[0], delta_range[1], n)
    Z_exact = (f * B) / np.maximum(d0 + deltas, 0.1)
    Z0      = (f * B) / d0
    slope   = -(f * B) / d0**2
    Z_lin   = Z0 + slope * deltas
    return {
        'deltas':  deltas,
        'Z_exact': Z_exact,
        'Z_lin':   Z_lin,
        'Z0':      Z0,
        'd0':      d0,
        'slope':   slope,
        'error_cm': np.abs(Z_lin - Z_exact) * 100
    }


def plot_full_error_analysis(
    f: float = 1735.0,
    B: float = 0.160,
    sigma_d: float = 0.5,
    Z_range: tuple = (0.3, 10.0),
    save_folder: str = "output",
    show: bool = True
):
    """
    Generate all error analysis plots for the paper.

    Figures:
        1. Absolute error σ_Z vs Z
        2. Relative error (σ_Z/Z) vs Z
        3. Baseline comparison
        4. Taylor linearization
        5. Sensitivity dZ/dd vs d
    """

    Z    = np.linspace(Z_range[0], Z_range[1], 400)
    d    = (f * B) / Z              # corresponding disparities
    sZ   = absolute_depth_error(Z, f, B, sigma_d)
    rZ   = relative_depth_error(Z,  f, B, sigma_d)
    sens = np.abs(depth_sensitivity(d, f, B))

    # ── Figure setup ────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=COLORS['bg'])
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                             hspace=0.45, wspace=0.35)

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(COLORS['bg'])
        ax.set_title(title, color=COLORS['accent2'], fontsize=10, pad=8)
        ax.set_xlabel(xlabel, color=COLORS['text'], fontsize=9)
        ax.set_ylabel(ylabel, color=COLORS['text'], fontsize=9)
        ax.tick_params(colors=COLORS['dimtext'], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS['panel'])
        ax.grid(True, alpha=0.12, color=COLORS['dimtext'])

    # ── Plot 1: Absolute error ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(Z, sZ * 100, color=COLORS['red'], linewidth=2.5)
    ax1.fill_between(Z, 0, sZ * 100, alpha=0.15, color=COLORS['red'])
    style_ax(ax1,
             f'Absolute Error σ_Z\n(σ_d={sigma_d}px)',
             'Depth Z (m)', 'σ_Z (cm)')
    # Annotations
    for z_mark in [1.0, 3.0, 5.0]:
        if Z_range[0] <= z_mark <= Z_range[1]:
            sz_mark = absolute_depth_error(z_mark, f, B, sigma_d) * 100
            ax1.annotate(f'{sz_mark:.2f}cm',
                          xy=(z_mark, sz_mark),
                          xytext=(z_mark + 0.3, sz_mark * 1.1),
                          color=COLORS['yellow'], fontsize=8,
                          arrowprops=dict(arrowstyle='->', color=COLORS['dimtext']))

    # ── Plot 2: Relative error ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(Z, rZ, color=COLORS['blue'], linewidth=2.5)
    ax2.fill_between(Z, 0, rZ, alpha=0.15, color=COLORS['blue'])
    style_ax(ax2,
             'Relative Error σ_Z/Z\n= (Z/fB)·σ_d',
             'Depth Z (m)', 'σ_Z/Z (%)')
    ax2.axhline(1.0, color=COLORS['yellow'], linestyle='--',
                linewidth=1.5, alpha=0.7, label='1% threshold')
    ax2.axhline(5.0, color=COLORS['red'],    linestyle='--',
                linewidth=1.5, alpha=0.7, label='5% threshold')
    ax2.legend(fontsize=8, facecolor=COLORS['panel'],
               labelcolor=COLORS['text'], framealpha=0.8)

    # ── Plot 3: Baseline comparison ──────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    baselines = [0.05, 0.10, 0.16, 0.30]
    pal       = [COLORS['red'], COLORS['yellow'],
                 COLORS['green'], COLORS['blue']]
    for b, c in zip(baselines, pal):
        sZ_b = absolute_depth_error(Z, f, b, sigma_d) * 100
        ax3.plot(Z, sZ_b, color=c, linewidth=2,
                 label=f'B = {b*100:.0f} cm')
    style_ax(ax3,
             'Effect of Baseline B\n(larger B → lower error)',
             'Depth Z (m)', 'σ_Z (cm)')
    ax3.legend(fontsize=8, facecolor=COLORS['panel'],
               labelcolor=COLORS['text'])

    # ── Plot 4: Taylor linearization ────────────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    d0   = (f * B) / 2.0    # linearize around Z=2m
    tay  = taylor_linearization(d0, f, B)

    ax4.plot(tay['deltas'], tay['Z_exact'], color=COLORS['blue'],
             linewidth=2.5, label='Exact: Z = fB/(d₀+Δd)')
    ax4.plot(tay['deltas'], tay['Z_lin'],   color=COLORS['red'],
             linewidth=2.0, linestyle='--',
             label=f'Linear: Z₀ + ({tay["slope"]:.4f})·Δd')
    ax4.axvline(0, color=COLORS['dimtext'], linestyle=':', alpha=0.5)
    ax4.axhline(tay['Z0'], color=COLORS['dimtext'], linestyle=':', alpha=0.5)
    style_ax(ax4,
             f'Taylor Linearization at d₀={d0:.1f}px  (Z₀={tay["Z0"]:.2f}m)',
             'Δd (pixels)', 'Depth Z (m)')
    ax4.legend(fontsize=9, facecolor=COLORS['panel'],
               labelcolor=COLORS['text'])

    # ── Plot 5: Sensitivity ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(d, sens, color=COLORS['green'], linewidth=2.5)
    ax5.fill_between(d, 0, sens, alpha=0.15, color=COLORS['green'])
    style_ax(ax5,
             'Sensitivity |dZ/dd|\n= fB/d²  (high near → unstable)',
             'Disparity d (px)', '|dZ/dd| (m/px)')

    # ── Super title ──────────────────────────────────────────
    fig.suptitle(
        f'Error Analysis  ·  f={f:.0f}px  ·  B={B*100:.1f}cm  '
        f'·  σ_d={sigma_d}px\n'
        f'Paper: Vlerësimi Gjeometrik i Thellësisë  |  UBT 2026  |  Mërgim Pirraku',
        color=COLORS['text'], fontsize=11, y=0.98
    )

    # ── Save ─────────────────────────────────────────────────
    os.makedirs(save_folder, exist_ok=True)
    out_path = os.path.join(save_folder, 'error_analysis_full.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    print(f"Saved → {out_path}")

    if show:
        plt.show()

    return fig


def print_error_table(
    f: float = 1735.0,
    B: float = 0.160,
    sigma_d: float = 0.5,
    depths: list = None
):
    """Print a formatted error table for the paper."""

    if depths is None:
        depths = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    print(f"\n{'═'*65}")
    print(f"  Error Analysis Table")
    print(f"  f={f:.0f}px  |  B={B*100:.1f}cm  |  σ_d={sigma_d}px")
    print(f"{'═'*65}")
    print(f"  {'Z (m)':>7} │ {'d (px)':>8} │ "
          f"{'σ_Z (cm)':>10} │ {'σ_Z/Z (%)':>11} │ {'|dZ/dd|':>10}")
    print(f"  {'─'*7}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*11}─┼─{'─'*10}")

    for Z in depths:
        d    = (f * B) / Z
        sZ   = absolute_depth_error(Z, f, B, sigma_d)
        rZ   = relative_depth_error(Z,  f, B, sigma_d)
        sens = abs(depth_sensitivity(d, f, B))
        print(f"  {Z:>7.2f} │ {d:>8.2f} │ "
              f"{sZ*100:>10.4f} │ {rZ:>11.4f} │ {sens:>10.5f}")

    print(f"{'═'*65}\n")


if __name__ == "__main__":
    # Default Middlebury parameters
    F = 1735.0   # pixels
    B = 0.160    # meters
    S = 0.5      # pixels

    print_error_table(f=F, B=B, sigma_d=S)
    plot_full_error_analysis(f=F, B=B, sigma_d=S,
                              save_folder="output", show=True)