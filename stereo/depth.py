import cv2
import numpy as np

def compute_depth_map(disparity, focal_length, baseline):
    """
    Core formula from Section 5.8:
    Z = (f * B) / d
    
    Parameters:
    -----------
    disparity   : 2D array of disparity values (pixels)
    focal_length: f in pixels (from rectified projection matrix P1[0,0])
    baseline    : B in meters (distance between cameras)
    
    Returns:
    --------
    depth_map   : Z in meters for each pixel
    """
    
    # Avoid division by zero
    safe_disparity = np.where(
        np.isnan(disparity) | (disparity <= 0),
        np.nan,
        disparity
    )
    
    # Z = f * B / d  (Equation from Section 5.8)
    depth_map = np.where(
        np.isnan(safe_disparity),
        np.nan,
        (focal_length * baseline) / safe_disparity
    )
    
    return depth_map


def compute_3d_points(disparity, Q_matrix):
    """
    Full 3D reconstruction using OpenCV's Q matrix.
    Computes X, Y, Z for every pixel.
    
    From Section 5.9:
    X = xL * Z / f
    Y = yL * Z / f
    Z = f * B / d
    
    Q matrix transforms (u, v, d) -> (X, Y, Z, W)
    """
    
    # Replace NaN with 0 for reprojectImageTo3D
    disp_clean = np.nan_to_num(disparity, nan=0.0)
    
    # Reproject to 3D space
    # Output shape: (H, W, 3) with (X, Y, Z) in real-world units
    points_3d = cv2.reprojectImageTo3D(disp_clean, Q_matrix)
    
    return points_3d


def get_depth_at_point(depth_map, x, y, window=5):
    """
    Get depth at pixel (x, y) with optional neighborhood averaging.
    Used for point-specific distance measurement.
    """
    
    h, w = depth_map.shape
    
    # Clamp coordinates
    x = max(window, min(w - window - 1, x))
    y = max(window, min(h - window - 1, y))
    
    # Extract neighborhood
    region = depth_map[y-window:y+window+1, x-window:x+window+1]
    valid_depths = region[~np.isnan(region)]
    
    if len(valid_depths) == 0:
        return None
    
    depth = np.median(valid_depths)  # Median is more robust than mean
    std = np.std(valid_depths)
    
    print(f"Depth at pixel ({x}, {y}): Z = {depth:.3f} m ± {std:.3f} m")
    
    return depth, std


def compute_depth_statistics(depth_map):
    """Statistical summary of the depth map"""
    
    valid = depth_map[~np.isnan(depth_map)]
    
    if len(valid) == 0:
        print("No valid depth values!")
        return None
    
    stats = {
        'min': np.min(valid),
        'max': np.max(valid),
        'mean': np.mean(valid),
        'median': np.median(valid),
        'std': np.std(valid),
        'valid_percentage': 100 * len(valid) / depth_map.size
    }
    
    print(f"\n=== Depth Map Statistics ===")
    print(f"Min depth:    {stats['min']:.3f} m")
    print(f"Max depth:    {stats['max']:.3f} m")
    print(f"Mean depth:   {stats['mean']:.3f} m")
    print(f"Median depth: {stats['median']:.3f} m")
    print(f"Std dev:      {stats['std']:.3f} m")
    print(f"Valid pixels: {stats['valid_percentage']:.1f}%")
    
    return stats