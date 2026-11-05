"""
3D Reconstruction Evaluation Metrics and Point Cloud I/O Utilities.
"""

from typing import Optional
import numpy as np


def _pairwise_min_distances(pts_a: np.ndarray, pts_b: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    """Compute minimum Euclidean distance from each point in pts_a (N, 3) to pts_b (M, 3)."""
    if len(pts_a) == 0 or len(pts_b) == 0:
        return np.array([np.inf], dtype=np.float32)

    min_dists = np.empty(len(pts_a), dtype=np.float32)
    for start in range(0, len(pts_a), chunk_size):
        chunk = pts_a[start : start + chunk_size]
        diff = chunk[:, None, :] - pts_b[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        min_dists[start : start + chunk_size] = np.sqrt(np.min(dist2, axis=-1))
    return min_dists


def chamfer_distance(pts_pred: np.ndarray, pts_gt: np.ndarray) -> float:
    """
    Symmetric L1 Chamfer Distance between predicted 3D point cloud (N, 3) and ground truth (M, 3).
    Lower is better.
    """
    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return float("inf")
    d_pred_to_gt = _pairwise_min_distances(pts_pred, pts_gt)
    d_gt_to_pred = _pairwise_min_distances(pts_gt, pts_pred)
    return float(0.5 * (np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred)))


def f_score_3d(pts_pred: np.ndarray, pts_gt: np.ndarray, threshold: float = 0.05) -> float:
    """
    Harmonic mean of 3D Precision (accuracy) and Recall (completeness) at distance threshold tau.
    Higher is better (in [0, 1]).
    """
    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return 0.0
    d_pred_to_gt = _pairwise_min_distances(pts_pred, pts_gt)
    d_gt_to_pred = _pairwise_min_distances(pts_gt, pts_pred)

    precision = float(np.mean(d_pred_to_gt < threshold))
    recall = float(np.mean(d_gt_to_pred < threshold))
    if precision + recall < 1e-8:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def volumetric_iou(occ_pred: np.ndarray, occ_gt: np.ndarray) -> float:
    """3D Volumetric Intersection-over-Union (IoU) between boolean voxel grids. Higher is better."""
    inter = np.logical_and(occ_pred, occ_gt).sum()
    union = np.logical_or(occ_pred, occ_gt).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def export_point_cloud_ply(
    filepath: str,
    points: np.ndarray,
    temperatures: Optional[np.ndarray] = None,
) -> None:
    """
    Save a 3D point cloud (N, 3) with optional thermal intensity values (N,) mapped to a
    false-color ironbow/inferno RGB palette as an ASCII .ply file (viewable in MeshLab/CloudCompare).
    """
    N = len(points)
    if temperatures is not None and len(temperatures) == N:
        t_norm = np.clip((temperatures - np.min(temperatures)) / (np.ptp(temperatures) + 1e-8), 0.0, 1.0)
        # Ironbow-like thermal colormap
        r = np.clip(255.0 * np.minimum(1.0, t_norm * 1.6), 0, 255).astype(np.uint8)
        g = np.clip(255.0 * np.clip((t_norm - 0.35) * 1.5, 0.0, 1.0), 0, 255).astype(np.uint8)
        b = np.clip(255.0 * np.sin(np.pi * t_norm) * 0.6, 0, 255).astype(np.uint8)
    else:
        r = np.full(N, 220, dtype=np.uint8)
        g = np.full(N, 120, dtype=np.uint8)
        b = np.full(N, 60, dtype=np.uint8)

    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for i in range(N):
        x, y, z = points[i]
        lines.append(f"{x:.5f} {y:.5f} {z:.5f} {int(r[i])} {int(g[i])} {int(b[i])}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

