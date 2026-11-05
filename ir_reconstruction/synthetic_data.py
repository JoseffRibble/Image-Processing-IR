"""
Synthetic Infrared (LWIR Thermal & NIR Active) Dataset Generator with Occlusions.

Models key physical properties of infrared imaging:
1. Stefan-Boltzmann thermal emission (I ~ emissivity * T^4) normalized to sensor range.
2. View-angle dependent directional emissivity drop-off at grazing angles.
3. Atmospheric attenuation over distance: tau(d) = exp(-beta * d).
4. Thermal blur (heat conduction / low optical MTF) and sensor noise.
5. Foreground occluders (e.g., cold vertical bars / foliage clutter) between camera and target.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class CameraPose:
    """Pinhole camera intrinsics and extrinsics (world-to-camera and camera-to-world)."""
    K: np.ndarray          # (3, 3) intrinsic matrix
    R_c2w: np.ndarray      # (3, 3) rotation from camera to world
    t_c2w: np.ndarray      # (3,) camera center in world coordinates
    width: int
    height: int

    def get_rays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ray origins (H, W, 3) and normalized world ray directions (H, W, 3)."""
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float32),
                           np.arange(self.height, dtype=np.float32))
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
        dirs_world = dirs_cam @ self.R_c2w.T
        norms = np.linalg.norm(dirs_world, axis=-1, keepdims=True) + 1e-8
        dirs_world = dirs_world / norms
        origins = np.broadcast_to(self.t_c2w.astype(np.float32), dirs_world.shape).copy()
        return origins, dirs_world


class SyntheticIRScene:
    """
    Defines a 3D composite thermal object (e.g., a heated industrial valve/torus + core sphere)
    inside [-1, 1]^3 with optional foreground occluder pillars/foliage.
    """

    def __init__(
        self,
        ambient_temp_k: float = 293.15,
        object_temp_k: float = 345.0,
        atm_beta: float = 0.05,
        num_occluders: int = 0,
        seed: int = 42,
    ):
        self.ambient_temp_k = ambient_temp_k
        self.object_temp_k = object_temp_k
        self.atm_beta = atm_beta
        self.num_occluders = num_occluders
        self.rng = np.random.default_rng(seed)

        # Generate random occluder cylinders arranged on a shell around the object (radius ~ 1.15)
        self.occluders = []
        if num_occluders > 0:
            angles = np.linspace(0, 2 * np.pi, num_occluders, endpoint=False)
            angles += self.rng.uniform(-0.15, 0.15, size=num_occluders)
            for ang in angles:
                cx = 1.1 * np.cos(ang)
                cy = 1.1 * np.sin(ang)
                radius = self.rng.uniform(0.10, 0.16)
                self.occluders.append((cx, cy, radius))

    def sdf_object(self, pts: np.ndarray) -> np.ndarray:
        """
        Signed Distance Function (SDF) of the target 3D object (composite shape:
        central sphere + horizontal torus + top cylinder).
        pts: (..., 3)
        Returns signed distance (negative inside object).
        """
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]

        # 1. Central heated sphere (radius 0.42)
        d_sphere = np.sqrt(x**2 + y**2 + z**2 + 1e-8) - 0.42

        # 2. Equatorial ring / flange (major radius 0.45, minor radius 0.13)
        q_xy = np.sqrt(x**2 + y**2 + 1e-8) - 0.45
        d_torus = np.sqrt(q_xy**2 + z**2 + 1e-8) - 0.13

        # 3. Vertical stem cylinder along Z (radius 0.22, height [-0.55, 0.55])
        d_cyl_r = np.sqrt(x**2 + y**2 + 1e-8) - 0.22
        d_cyl_z = np.abs(z) - 0.55
        d_cyl = np.maximum(d_cyl_r, d_cyl_z)

        return np.minimum(d_sphere, np.minimum(d_torus, d_cyl))

    def sdf_occluders(self, pts: np.ndarray) -> np.ndarray:
        """Signed Distance Function of foreground occluders (cold vertical pillars/bars)."""
        if not self.occluders:
            return np.full(pts.shape[:-1], 1e6, dtype=np.float32)

        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        d_min = np.full(pts.shape[:-1], 1e6, dtype=np.float32)
        for cx, cy, r in self.occluders:
            d_pillar = np.maximum(np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1e-8) - r, np.abs(z) - 0.9)
            d_min = np.minimum(d_min, d_pillar)
        return d_min

    def temperature_field(self, pts: np.ndarray) -> np.ndarray:
        """
        Spatial temperature distribution T(x, y, z) in Kelvin on the object.
        Creates a smooth vertical & radial thermal gradient (simulating heat conduction).
        """
        r2 = pts[..., 0] ** 2 + pts[..., 1] ** 2
        z = pts[..., 2]
        # Core is hottest, cooling slightly toward the outer flange and top
        delta_t = (self.object_temp_k - self.ambient_temp_k) * (
            1.0 - 0.25 * r2 + 0.15 * np.sin(3.0 * z)
        )
        return self.ambient_temp_k + np.clip(delta_t, 10.0, 80.0)

    def ground_truth_voxel_grid(self, grid_res: int = 48, bounds: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Return (coords (N,3), occupancy_grid (R,R,R) bool) for ground-truth object."""
        lin = np.linspace(-bounds, bounds, grid_res, dtype=np.float32)
        X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
        pts = np.stack([X, Y, Z], axis=-1)
        occ = self.sdf_object(pts) <= 0.0
        return pts, occ

    def ground_truth_surface_points(self, num_points: int = 4000, bounds: float = 0.75) -> np.ndarray:
        """Extract surface points where |SDF(p)| < threshold."""
        pts, _ = self.ground_truth_voxel_grid(grid_res=80, bounds=bounds)
        sdf_vals = np.abs(self.sdf_object(pts))
        surf_pts = pts[sdf_vals < 0.02]
        if len(surf_pts) > num_points:
            idx = self.rng.choice(len(surf_pts), size=num_points, replace=False)
            surf_pts = surf_pts[idx]
        return surf_pts


def generate_turntable_cameras(
    num_views: int = 16,
    radius: float = 2.4,
    elevation_deg: float = 25.0,
    width: int = 64,
    height: int = 64,
    fov_deg: float = 45.0,
) -> List[CameraPose]:
    """Generate calibrated multi-view cameras orbiting the origin."""
    cameras = []
    focal = 0.5 * width / np.tan(0.5 * np.deg2rad(fov_deg))
    K = np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    elev = np.deg2rad(elevation_deg)
    for i in range(num_views):
        azim = 2.0 * np.pi * i / num_views
        # Vary elevation slightly to give richer 3D multi-view coverage
        el = elev + np.deg2rad(10.0 * np.sin(2 * azim))
        cam_pos = np.array([
            radius * np.cos(el) * np.cos(azim),
            radius * np.cos(el) * np.sin(azim),
            radius * np.sin(el),
        ], dtype=np.float32)

        forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up_world)
        right /= np.linalg.norm(right) + 1e-8
        down = np.cross(forward, right)

        R_c2w = np.stack([right, down, forward], axis=1)
        cameras.append(CameraPose(K=K, R_c2w=R_c2w, t_c2w=cam_pos, width=width, height=height))
    return cameras


def render_multiview_ir_dataset(
    scene: SyntheticIRScene,
    cameras: List[CameraPose],
    num_ray_samples: int = 96,
    near: float = 1.0,
    far: float = 3.8,
    noise_std: float = 0.01,
) -> dict:
    """
    Render multi-view thermal infrared images with Stefan-Boltzmann emission,
    directional emissivity, atmospheric attenuation, and optional occluders.

    Returns dict with:
        - 'thermal_images': (V, H, W) normalized thermal intensity in [0, 1]
        - 'depth_maps': (V, H, W) ground truth ray termination depth
        - 'occluder_masks': (V, H, W) boolean mask where foreground occluders block the ray
        - 'object_masks': (V, H, W) visible object mask
    """
    t_vals = np.linspace(near, far, num_ray_samples, dtype=np.float32)
    thermal_images = []
    depth_maps = []
    occluder_masks = []
    object_masks = []

    # Normalization constants (Stefan-Boltzmann T^4 range)
    t_min4 = scene.ambient_temp_k ** 4
    t_max4 = (scene.object_temp_k + 15.0) ** 4

    for cam in cameras:
        origins, dirs = cam.get_rays()  # (H, W, 3)
        # Sample points along rays: (H, W, S, 3)
        pts = origins[..., None, :] + dirs[..., None, :] * t_vals[None, None, :, None]

        d_obj = scene.sdf_object(pts)       # (H, W, S)
        d_occ = scene.sdf_occluders(pts)    # (H, W, S)

        hit_obj_all = d_obj <= 0.0
        hit_occ_all = d_occ <= 0.0

        # Find first hit index along each ray
        has_obj = np.any(hit_obj_all, axis=-1)
        has_occ = np.any(hit_occ_all, axis=-1)

        idx_obj = np.where(has_obj, np.argmax(hit_obj_all, axis=-1), num_ray_samples)
        idx_occ = np.where(has_occ, np.argmax(hit_occ_all, axis=-1), num_ray_samples)

        occ_blocks = has_occ & (idx_occ < idx_obj)
        obj_visible = has_obj & (~occ_blocks)

        # Compute hit coordinates and surface normals via finite differences
        idx_first = np.clip(idx_obj, 0, num_ray_samples - 1)
        H, W = origins.shape[:2]
        rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        hit_pts = pts[rows, cols, idx_first]
        hit_dist = t_vals[idx_first]

        # Finite-difference normal on object SDF for directional emissivity
        eps = 1e-3
        nx = scene.sdf_object(hit_pts + np.array([eps, 0, 0], dtype=np.float32)) - scene.sdf_object(hit_pts - np.array([eps, 0, 0], dtype=np.float32))
        ny = scene.sdf_object(hit_pts + np.array([0, eps, 0], dtype=np.float32)) - scene.sdf_object(hit_pts - np.array([0, eps, 0], dtype=np.float32))
        nz = scene.sdf_object(hit_pts + np.array([0, 0, eps], dtype=np.float32)) - scene.sdf_object(hit_pts - np.array([0, 0, eps], dtype=np.float32))
        normals = np.stack([nx, ny, nz], axis=-1)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8

        cos_theta = np.clip(-np.sum(normals * dirs, axis=-1), 0.0, 1.0)
        # Directional emissivity in LWIR (drops off near grazing angles)
        emissivity = 0.92 * (1.0 - 0.25 * (1.0 - cos_theta) ** 3)

        # Temperature & Stefan-Boltzmann radiance
        temp_k = scene.temperature_field(hit_pts)
        norm_rad = np.clip((temp_k ** 4 - t_min4) / (t_max4 - t_min4), 0.0, 1.0)

        # Atmospheric transmission: tau(d) = exp(-beta * d)
        tau = np.exp(-scene.atm_beta * hit_dist)

        img = np.zeros((H, W), dtype=np.float32)
        img[obj_visible] = emissivity[obj_visible] * norm_rad[obj_visible] * tau[obj_visible]

        # Occluders are at near-ambient temperature (e.g., cold foliage/structures -> low intensity ~ 0.04)
        img[occ_blocks] = 0.04

        if noise_std > 0:
            img += scene.rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)

        depth = np.where(obj_visible, hit_dist, 0.0)
        thermal_images.append(img)
        depth_maps.append(depth)
        occluder_masks.append(occ_blocks)
        object_masks.append(obj_visible)

    return {
        "thermal_images": np.stack(thermal_images, axis=0),
        "depth_maps": np.stack(depth_maps, axis=0),
        "occluder_masks": np.stack(occluder_masks, axis=0),
        "object_masks": np.stack(object_masks, axis=0),
    }


def render_nir_photometric_stereo_dataset(
    grid_size: int = 64,
    num_lights: int = 8,
    occlude_lights: bool = True,
    seed: int = 42,
) -> dict:
    """
    Generate a synthetic Near-Infrared (NIR) Photometric Stereo dataset of a 3D surface Z(x,y)
    under multiple calibrated NIR LED light directions, with optional shadow/occluder corruption.
    """
    rng = np.random.default_rng(seed)
    lin = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    X, Y = np.meshgrid(lin, lin)

    # Smooth 3D dome + double-peak surface inside radius < 0.75
    r2 = X**2 + Y**2
    mask = r2 < 0.56
    Z_gt = np.zeros_like(X)
    Z_gt[mask] = (
        0.38 * np.sqrt(np.clip(0.56 - r2[mask], 0.0, None))
        + 0.08 * np.cos(4.0 * X[mask]) * np.cos(4.0 * Y[mask])
    )

    # Compute ground-truth surface gradients (p = dZ/dx, q = dZ/dy) and normals
    dx = lin[1] - lin[0]
    q_gt, p_gt = np.gradient(Z_gt, dx, dx)
    normals_gt = np.stack([-p_gt, -q_gt, np.ones_like(Z_gt)], axis=-1)
    normals_gt /= np.linalg.norm(normals_gt, axis=-1, keepdims=True) + 1e-8
    normals_gt[~mask] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Ring of NIR LEDs at 45-degree slant angle
    angles = np.linspace(0, 2 * np.pi, num_lights, endpoint=False)
    slant = np.deg2rad(42.0)
    light_dirs = np.stack([
        np.sin(slant) * np.cos(angles),
        np.sin(slant) * np.sin(angles),
        np.cos(slant) * np.ones_like(angles),
    ], axis=-1).astype(np.float32)

    albedo = 0.88  # Uniform NIR subsurface albedo (key advantage of NIR over RGB!)
    images = []
    for k, l_dir in enumerate(light_dirs):
        n_dot_l = np.clip(np.sum(normals_gt * l_dir[None, None, :], axis=-1), 0.0, 1.0)
        img = albedo * n_dot_l * mask

        if occlude_lights and (k % 3 == 0):
            # Simulate an occluder bar casting a shadow across part of the object in some light views
            bar_center = -0.25 + 0.2 * (k // 3)
            shadow_band = np.abs(X * np.cos(angles[k]) + Y * np.sin(angles[k]) - bar_center) < 0.14
            img[shadow_band] = 0.01

        img += rng.normal(0.0, 0.008, size=img.shape).astype(np.float32)
        images.append(np.clip(img, 0.0, 1.0))

    return {
        "nir_images": np.stack(images, axis=0),   # (L, H, W)
        "light_dirs": light_dirs,                 # (L, 3)
        "mask": mask,                             # (H, W)
        "Z_gt": Z_gt,                             # (H, W)
        "normals_gt": normals_gt,                 # (H, W, 3)
        "X": X,
        "Y": Y,
    }

