import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import cv2


# ── Helpers ──────────────────────────────────────────────────

def make_synthetic_pair(width=640, height=480, disparity_shift=30):
    """
    Create a simple rectified stereo pair:
    right image = left image shifted left by `disparity_shift` px.
    """
    left = np.zeros((height, width), dtype=np.uint8)

    # Draw some geometry
    cv2.rectangle(left, (100, 100), (250, 350), 200, -1)
    cv2.circle(left,    (400, 240), 80,          150, -1)
    cv2.rectangle(left, (500, 150), (580, 400),  180, -1)

    right = np.roll(left, -disparity_shift, axis=1)
    right[:, -disparity_shift:] = 0   # blank out rolled edge

    return (cv2.cvtColor(left,  cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(right, cv2.COLOR_GRAY2BGR))


# ── Tests ────────────────────────────────────────────────────

def test_depth_formula():
    """Z = f·B / d"""
    f, B, d = 1735.0, 0.16, 34.7
    Z_expected = f * B / d
    assert abs(Z_expected - 8.0) < 0.1, f"Expected ~8.0 m, got {Z_expected:.3f}"
    print("  ✓ test_depth_formula")


def test_error_formulas():
    from analysis.error_analysis import (
        absolute_depth_error,
        relative_depth_error,
        depth_sensitivity,
    )
    f, B, Z, sigma_d = 1735.0, 0.16, 2.0, 0.5

    abs_err = absolute_depth_error(Z, f, B, sigma_d)
    rel_err = relative_depth_error(Z, f, B, sigma_d)
    d       = (f * B) / Z
    sens    = depth_sensitivity(d, f, B)

    assert abs_err > 0,    "Absolute error should be positive"
    assert 0 < rel_err < 1, "Relative error should be < 100% at 2m"
    assert sens < 0,        "Sensitivity dZ/dd should be negative"
    print(f"  ✓ test_error_formulas  "
          f"(σ_Z={abs_err*100:.3f}cm, σ_Z/Z={rel_err*100:.3f}%)")


def test_taylor_linearization():
    from analysis.error_analysis import taylor_linearization
    data = taylor_linearization(d0=30.0, focal_length=1735.0, baseline=0.16)

    assert 'Z_exact' in data
    assert 'Z_lin'   in data
    # At Δd=0 both should equal Z0
    mid = len(data['deltas']) // 2
    assert abs(data['Z_exact'][mid] - data['Z0']) < 0.001
    assert abs(data['Z_lin'][mid]   - data['Z0']) < 0.001
    print("  ✓ test_taylor_linearization")


def test_pipeline_disparity():
    """Full pipeline smoke-test with synthetic images"""
    # Import inline to avoid circular issues
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from stereo_vision_app import StereoPipeline

    p = StereoPipeline()
    p.img_left, p.img_right = make_synthetic_pair()

    # No calibration → use as pre-rectified
    p.img_left_rect  = p.img_left
    p.img_right_rect = p.img_right

    ok_d = p.compute_disparity()
    assert ok_d, "Disparity computation failed"
    assert p.disparity is not None
    assert p.disparity.shape[:2] == p.img_left.shape[:2]
    print(f"  ✓ test_pipeline_disparity  "
          f"(range {np.nanmin(p.disparity):.1f}–{np.nanmax(p.disparity):.1f} px)")

    ok_z = p.compute_depth()
    assert ok_z, "Depth computation failed"
    assert p.depth_map is not None
    print(f"  ✓ test_pipeline_depth  "
          f"(mean {np.nanmean(p.depth_map):.2f} m)")


def test_stats():
    from stereo_vision_app import StereoPipeline
    p = StereoPipeline()
    p.img_left, p.img_right = make_synthetic_pair()
    p.img_left_rect  = p.img_left
    p.img_right_rect = p.img_right
    p.compute_disparity()
    p.compute_depth()

    stats = p.get_depth_stats()
    assert stats, "Stats should not be empty after depth computation"
    assert stats['min'] <= stats['median'] <= stats['max']
    assert 0 < stats['valid'] <= 100
    print(f"  ✓ test_stats  (valid={stats['valid']:.1f}%)")


# ── Runner ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Stereo Vision Pipeline — Smoke Tests")
    print("=" * 50)

    tests = [
        test_depth_formula,
        test_error_formulas,
        test_taylor_linearization,
        test_pipeline_disparity,
        test_stats,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed}/{len(tests)} passed")
    print('=' * 50)