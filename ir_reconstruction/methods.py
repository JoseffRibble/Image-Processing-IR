"""
3D Reconstruction Algorithms Adapted for Infrared (LWIR Thermal & NIR) and Occlusion.

Implements 4 complementary paradigms:
1. Occlusion-Aware Thermal Space Carving (`reconstruct_thermal_space_carving`)
2. Airborne Optical Sectioning / Synthetic Aperture Focusing (`reconstruct_aos_focal_stack`)
   - Ref: Schedl, Kurmi, & Bimber, Nature Machine Intelligence (2020)
3. Robust Near-Infrared Photometric Stereo (`reconstruct_ir_photometric_stereo`)
   - Ref: Hansen et al., MVA (2010) + IRLS robust shadow/occlusion rejection
4. Physics-Informed Robust Thermal Radiance Field (`reconstruct_thermal_radiance_field`)
   - Ref: Ye et al. (Thermal-NeRF, 2024), Chen et al. (Thermal3D-GS, ECCV 2024),
          and Sabour et al. (RobustNeRF, CVPR 2023)
"""

from typing import Dict, List, Tuple
import numpy as np
from .synthetic_data import CameraPose


def _project_points_to_camera(pts: np.ndarray, cam: CameraPose) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D world points (..., 3) into a camera image plane.
    Returns (u, v, z_cam) of shape (...).
    """
    R_w2c = cam.R_c2w.T
    pts_cam = (pts - cam.t_c2w) @ R_w2c.T
    z_cam = pts_cam[..., 2]
    z_safe = np.where(z_cam > 1e-4, z_cam, 1e-4)

    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    u = fx * (pts_cam[..., 0] / z_safe) + cx
    v = fy * (pts_cam[..., 1] / z_safe) + cy
    return u, v, z_cam


def _bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample 2D image (H, W) at coordinates (u, v). Returns (sampled_vals, in_bounds_mask)."""
    H, W = img.shape
    in_bounds = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    u_c = np.clip(u, 0, W - 1.001)
    v_c = np.clip(v, 0, H - 1.001)

    u0 = np.floor(u_c).astype(np.int32)
    v0 = np.floor(v_c).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    du = u_c - u0
    dv = v_c - v0

    val = (
        (1 - du) * (1 - dv) * img[v0, u0]
        + du * (1 - dv) * img[v0, u1]
        + (1 - du) * dv * img[v1, u0]
        + du * dv * img[v1, u1]
    )
    return val, in_bounds


# =============================================================================
# Method 1: Occlusion-Aware Thermal Space Carving
# =============================================================================

def reconstruct_thermal_space_carving(
    thermal_images: np.ndarray,
    cameras: List[CameraPose],
    grid_res: int = 48,
    bounds: float = 0.8,
    thermal_threshold: float = 0.15,
    min_view_consistency: float = 0.75,
) -> Dict[str, np.ndarray]:
    """
    Reconstruct a 3D voxel grid and surface point cloud by projecting 3D voxels into
    multi-view thermal silhouettes.

    Why standard Space Carving fails under occlusion:
        Classical Visual Hull requires 100% of views (`min_view_consistency = 1.0`) to see
        foreground thermal intensity; a single cold occluder carves away the object.
        By relaxing `min_view_consistency` (e.g., 0.70 - 0.80), voxels that are occluded
        in a subset of views are preserved via multi-view consensus voting.
    """
    lin = np.linspace(-bounds, bounds, grid_res, dtype=np.float32)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1)

    vote_count = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)
    valid_views = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)
    temp_accum = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)

    for img, cam in zip(thermal_images, cameras):
        u, v, z_cam = _project_points_to_camera(pts, cam)
        val, in_bounds = _bilinear_sample(img, u, v)
        valid = in_bounds & (z_cam > 0.1)

        is_hot = valid & (val >= thermal_threshold)
        vote_count += is_hot.astype(np.float32)
        valid_views += valid.astype(np.float32)
        temp_accum += np.where(is_hot, val, 0.0)

    consistency = vote_count / np.maximum(valid_views, 1.0)
    occupancy = (consistency >= min_view_consistency) & (valid_views >= len(cameras) * 0.5)
    mean_thermal = temp_accum / np.maximum(vote_count, 1.0)

    # Extract boundary surface points (occupied voxels with at least one unoccupied 6-neighbor)
    padded = np.pad(occupancy, 1, mode="constant", constant_values=False)
    eroded = (
        padded[1:-1, 1:-1, 1:-1]
        & padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    surface_mask = occupancy & (~eroded)
    surface_pts = pts[surface_mask]
    surface_temps = mean_thermal[surface_mask]

    return {
        "occupancy": occupancy,
        "surface_points": surface_pts,
        "surface_temperatures": surface_temps,
        "consistency_grid": consistency,
    }


# =============================================================================
# Method 2: Airborne Optical Sectioning (AOS) / Synthetic Aperture Focal Stack
# =============================================================================

def reconstruct_aos_focal_stack(
    thermal_images: np.ndarray,
    cameras: List[CameraPose],
    grid_res: int = 48,
    bounds: float = 0.8,
    thermal_threshold: float = 0.25,
    trim_fraction: float = 0.35,
) -> Dict[str, np.ndarray]:
    """
    Adapts Airborne Optical Sectioning (Schedl et al., Nature Machine Intelligence 2020)
    to 3D volumetric thermal reconstruction through heavy occlusions.

    Principle:
        When back-projecting multi-view rays onto a 3D voxel x:
        - Empty-space voxels observe cold background (near 0) in views that miss the object,
          whereas occluded object voxels observe foreground occluders (which cause localized
          intensity drops in a subset of views) while maintaining strong, low-variance thermal
          emission across the remaining unoccluded views.
        - By combining an adaptive trimmed synthetic-aperture integral with inter-view
          radiometric coherence (inverse coefficient of variation), foreground occluders
          defocus and vanish while the heated 3D object surface is reconstructed.
    """
    lin = np.linspace(-bounds, bounds, grid_res, dtype=np.float32)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1)

    V = len(cameras)
    samples = np.zeros((V, grid_res, grid_res, grid_res), dtype=np.float32)
    is_hot_ray = np.zeros((V, grid_res, grid_res, grid_res), dtype=bool)
    is_occluder_ray = np.zeros((V, grid_res, grid_res, grid_res), dtype=bool)

    for idx, (img, cam) in enumerate(zip(thermal_images, cameras)):
        u, v, z_cam = _project_points_to_camera(pts, cam)
        val, in_bounds = _bilinear_sample(img, u, v)
        valid = in_bounds & (z_cam > 0.1)

        # Compensate for atmospheric distance attenuation: exp(+beta * d)
        val_comp = val * np.exp(0.05 * np.clip(z_cam, 0.5, 4.0))
        samples[idx] = np.where(valid, val_comp, 0.0)
        is_hot_ray[idx] = valid & (val >= 0.15)
        # Cold foreground occluders have a faint ambient thermal signature (~0.025 - 0.08)
        # distinct from pure open background (~0.0)
        is_occluder_ray[idx] = valid & (val >= 0.022) & (val < 0.15)

    # Sort along view axis to reject occluded views (up to trim_fraction)
    sorted_samples = np.sort(samples, axis=0)
    k_start = int(np.floor(V * trim_fraction))
    unoccluded_slice = sorted_samples[k_start:, ...]

    robust_mean = np.mean(unoccluded_slice, axis=0)
    robust_std = np.std(unoccluded_slice, axis=0)

    # Fraction of views seeing either the heated target OR a detected foreground occluder
    effective_support = np.mean(is_hot_ray | is_occluder_ray, axis=0)
    hot_support = np.mean(is_hot_ray, axis=0)

    # Coherence score penalizes high variance across the unoccluded aperture
    coherence_score = robust_mean * np.exp(-3.5 * robust_std)
    occupancy = (
        (effective_support >= 0.90)
        & (hot_support >= (1.0 - trim_fraction - 0.10))
        & (coherence_score >= thermal_threshold)
    )

    padded = np.pad(occupancy, 1, mode="constant", constant_values=False)
    eroded = (
        padded[1:-1, 1:-1, 1:-1]
        & padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    surface_mask = occupancy & (~eroded)

    return {
        "occupancy": occupancy,
        "surface_points": pts[surface_mask],
        "surface_temperatures": robust_mean[surface_mask],
        "focal_volume": coherence_score,
    }


# =============================================================================
# Method 3: Robust Near-Infrared (NIR) Photometric Stereo
# =============================================================================

def _frankot_chellappa_integrate(p: np.ndarray, q: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Integrate surface gradients (p = dZ/dx, q = dZ/dy) into a 3D heightmap Z(x,y)
    in the Fourier domain using the Frankot-Chellappa projection onto integrable fields.
    """
    H, W = p.shape
    wx = np.fft.fftfreq(W, d=dx) * 2.0 * np.pi
    wy = np.fft.fftfreq(H, d=dx) * 2.0 * np.pi
    WX, WY = np.meshgrid(wx, wy)

    Fp = np.fft.fft2(p)
    Fq = np.fft.fft2(q)

    denom = WX**2 + WY**2
    denom[0, 0] = 1.0  # avoid division by zero at DC component
    FZ = (-1j * WX * Fp - 1j * WY * Fq) / denom
    FZ[0, 0] = 0.0

    Z = np.real(np.fft.ifft2(FZ)).astype(np.float32)
    return Z


def reconstruct_ir_photometric_stereo(
    nir_images: np.ndarray,
    light_dirs: np.ndarray,
    mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    irls_iters: int = 6,
) -> Dict[str, np.ndarray]:
    """
    Reconstruct a dense 3D surface Z(x, y) and surface normals from Near-Infrared (NIR)
    Photometric Stereo images using Iteratively Reweighted Least Squares (IRLS) with a
    Charbonnier/Huber weight to reject cast shadows and partial light occlusions.
    """
    L, H, W = nir_images.shape
    I_flat = nir_images.reshape(L, -1).T  # (N_pix, L)
    N_pix = I_flat.shape[0]

    # Solve I = S @ g for scaled normal g = albedo * n using IRLS
    g_est = np.zeros((N_pix, 3), dtype=np.float32)
    weights = np.ones((N_pix, L), dtype=np.float32)
    if irls_iters > 1:
        # Downweight very dark pixels initially (cast shadows / light occlusions)
        weights = np.where(I_flat < 0.03, 0.02, 1.0).astype(np.float32)

    for it in range(irls_iters):
        # Weighted least squares per pixel: (S^T W S) g = S^T W I
        SW = light_dirs[None, :, :] * weights[:, :, None]          # (N_pix, L, 3)
        lhs = np.matmul(SW.transpose(0, 2, 1), light_dirs[None, :, :])  # (N_pix, 3, 3)
        lhs += 1e-4 * np.eye(3, dtype=np.float32)[None, :, :]
        rhs = np.matmul(SW.transpose(0, 2, 1), I_flat[:, :, None])      # (N_pix, 3, 1)
        g_est = np.linalg.solve(lhs, rhs)[..., 0]                       # (N_pix, 3)

        if it + 1 < irls_iters:
            # Compute residuals and update robust Cauchy/Geman-McClure weights
            pred = g_est @ light_dirs.T  # (N_pix, L)
            res = np.abs(I_flat - pred)
            weights = 1.0 / ((1.0 + (res / 0.02) ** 2) ** 2)
            weights = np.where(I_flat < 0.025, 0.005, weights).astype(np.float32)

    norms = np.linalg.norm(g_est, axis=-1, keepdims=True) + 1e-8
    normals = (g_est / norms).reshape(H, W, 3)
    normals[~mask] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    nz = np.clip(normals[..., 2], 0.15, 1.0)
    p = -normals[..., 0] / nz
    q = -normals[..., 1] / nz
    p[~mask] = 0.0
    q[~mask] = 0.0

    dx = float(X[0, 1] - X[0, 0])
    Z_rec = _frankot_chellappa_integrate(p, q, dx=dx)
    # Zero-reference at the mask boundary
    Z_rec -= np.min(Z_rec[mask])
    Z_rec[~mask] = 0.0

    pts_3d = np.stack([X[mask], Y[mask], Z_rec[mask]], axis=-1)
    return {
        "Z_reconstructed": Z_rec,
        "normals_reconstructed": normals,
        "surface_points": pts_3d,
    }


# =============================================================================
# Method 4: Physics-Informed Robust Thermal Radiance Field (Thermal-NeRF / 3DGS style)
# =============================================================================

def reconstruct_thermal_radiance_field(
    thermal_images: np.ndarray,
    cameras: List[CameraPose],
    grid_res: int = 48,
    bounds: float = 0.8,
    num_iters: int = 35,
    lr: float = 0.40,
    tv_weight: float = 0.03,
    robust_delta: float = 0.06,
    atm_beta: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Physics-Informed Volumetric Thermal Radiance Field Optimization.

    Combines key ideas from:
    1. Thermal3D-GS / Thermal-NeRF: Models atmospheric attenuation exp(-beta * d) and
       3D heat-conduction smoothness (3D Laplacian regularizer derived from steady-state
       Fourier heat conduction nabla^2 T = 0).
    2. RobustNeRF (Sabour et al., CVPR 2023): Uses an iteratively reweighted robust
       Geman-McClure kernel with multi-view quantile consensus so that foreground cold
       occluders (where a ray reads ambient occluder intensity while the multi-view
       consensus is hot) are rejected as distractors without carving away the 3D object.
    """
    lin = np.linspace(-bounds, bounds, grid_res, dtype=np.float32)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1)

    V = len(cameras)
    proj_z = np.zeros((V, grid_res, grid_res, grid_res), dtype=np.float32)
    obs_vals = np.zeros((V, grid_res, grid_res, grid_res), dtype=np.float32)
    valid_mask = np.zeros((V, grid_res, grid_res, grid_res), dtype=bool)

    for i, (img, cam) in enumerate(zip(thermal_images, cameras)):
        u, v, z_cam = _project_points_to_camera(pts, cam)
        val, in_b = _bilinear_sample(img, u, v)
        proj_z[i] = z_cam
        obs_vals[i] = val
        valid_mask[i] = in_b & (z_cam > 0.1)

    # Atmospheric transmission factor per view-voxel pair
    tau = np.exp(-atm_beta * np.clip(proj_z, 0.5, 4.0))
    obs_comp = obs_vals / np.maximum(tau, 1e-4)

    # Identify candidate visual hull where rays either see hot object OR cold occluder (val >= 0.022)
    # Empty background rays (val < 0.022) strongly carve free space
    empty_ray_fraction = np.mean(valid_mask & (obs_vals < 0.022), axis=0)
    hull_prior = empty_ray_fraction <= 0.12

    # Initialize thermal radiance field from upper-quartile consensus inside hull prior
    q75 = np.percentile(obs_comp, 75, axis=0).astype(np.float32)
    field = np.where(hull_prior, q75, 0.0).astype(np.float32)

    loss_history = []
    for it in range(num_iters):
        pred = field[None, ...] * tau  # (V, R, R, R)
        residual = pred - obs_vals

        # Geman-McClure robust weight: w(r) = 1 / (1 + (r / delta)^2)^2
        abs_res = np.abs(residual)
        robust_w = 1.0 / ((1.0 + (abs_res / robust_delta) ** 2) ** 2)

        # RobustNeRF distractor gate: if a ray sees a cold occluder (0.022 <= obs < 0.15)
        # while the 75th-percentile multi-view consensus is hot (> 0.25), downweight that ray
        is_occluder_distractor = (obs_vals >= 0.022) & (obs_vals < 0.15) & (q75[None, ...] > 0.25)
        # Conversely, genuinely empty background rays (obs < 0.022) get high weight to carve free space
        is_empty_bg = obs_vals < 0.022
        robust_w = np.where(is_occluder_distractor, 0.01 * robust_w, robust_w)
        robust_w = np.where(is_empty_bg, 2.5, robust_w)
        robust_w = np.where(valid_mask, robust_w, 0.0)

        grad_data = np.sum(robust_w * residual * tau, axis=0) / np.maximum(
            np.sum(robust_w, axis=0), 1e-4
        )

        # 3D Fourier Heat Conduction Laplacian prior (nabla^2 T)
        padded = np.pad(field, 1, mode="edge")
        laplacian = (
            padded[:-2, 1:-1, 1:-1]
            + padded[2:, 1:-1, 1:-1]
            + padded[1:-1, :-2, 1:-1]
            + padded[1:-1, 2:, 1:-1]
            + padded[1:-1, 1:-1, :-2]
            + padded[1:-1, 1:-1, 2:]
            - 6.0 * field
        )

        field = field - lr * (grad_data - tv_weight * laplacian)
        field = np.where(hull_prior, np.clip(field, 0.0, 1.0), 0.0)
        loss_history.append(float(np.mean(robust_w * (residual**2))))

    occupancy = field >= 0.25
    padded_occ = np.pad(occupancy, 1, mode="constant", constant_values=False)
    eroded = (
        padded_occ[1:-1, 1:-1, 1:-1]
        & padded_occ[:-2, 1:-1, 1:-1]
        & padded_occ[2:, 1:-1, 1:-1]
        & padded_occ[1:-1, :-2, 1:-1]
        & padded_occ[1:-1, 2:, 1:-1]
        & padded_occ[1:-1, 1:-1, :-2]
        & padded_occ[1:-1, 1:-1, 2:]
    )
    surface_mask = occupancy & (~eroded)

    return {
        "occupancy": occupancy,
        "surface_points": pts[surface_mask],
        "surface_temperatures": field[surface_mask],
        "radiance_grid": field,
        "loss_history": np.array(loss_history, dtype=np.float32),
    }

