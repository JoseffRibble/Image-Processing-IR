"""
Benchmark and Test Suite for 3D Object Reconstruction Using Infrared Images.

Runs and compares:
1. Classical Visual Hull / Space Carving (`min_view_consistency=1.0`) vs. Occlusion-Aware Voting (`0.68`)
2. Airborne Optical Sectioning (AOS) Synthetic Aperture Focal Stack (Schedl et al., 2020)
3. Physics-Informed Robust Thermal Radiance Field (Thermal-NeRF / Thermal3D-GS + RobustNeRF)
4. Robust Near-Infrared (NIR) Photometric Stereo with IRLS Shadow/Occlusion Rejection

Usage:
    python3 test_pipeline.py
"""

import os
from typing import Tuple
import numpy as np

from ir_reconstruction import (
    SyntheticIRScene,
    generate_turntable_cameras,
    render_multiview_ir_dataset,
    render_nir_photometric_stereo_dataset,
    reconstruct_thermal_space_carving,
    reconstruct_aos_focal_stack,
    reconstruct_ir_photometric_stereo,
    reconstruct_thermal_radiance_field,
    chamfer_distance,
    f_score_3d,
    volumetric_iou,
    export_point_cloud_ply,
)


def run_multiview_thermal_benchmark(
    num_occluders: int = 0,
    grid_res: int = 44,
    export_dir: str = "outputs",
) -> dict:
    """Evaluate multi-view thermal 3D reconstruction methods with or without occlusions."""
    tag = "unoccluded" if num_occluders == 0 else f"occluded_{num_occluders}_pillars"
    print(f"\n{'=' * 78}")
    print(f"Scenario: Multi-View LWIR Thermal Reconstruction ({tag})")
    print(f"{'=' * 78}")

    scene = SyntheticIRScene(num_occluders=num_occluders, seed=42)
    cameras = generate_turntable_cameras(num_views=18, width=64, height=64)
    dataset = render_multiview_ir_dataset(scene, cameras, num_ray_samples=84)

    _, occ_gt = scene.ground_truth_voxel_grid(grid_res=grid_res, bounds=0.8)
    pts_gt = scene.ground_truth_surface_points(num_points=2500, bounds=0.8)

    occ_ratio = float(np.mean(dataset["occluder_masks"])) * 100.0
    print(f"Rendered {len(cameras)} thermal views (64x64) | Mean pixel occlusion rate: {occ_ratio:.1f}%")

    methods = {
        "1a. Strict Space Carving (Visual Hull)": lambda: reconstruct_thermal_space_carving(
            dataset["thermal_images"],
            cameras,
            grid_res=grid_res,
            min_view_consistency=0.98,
        ),
        "1b. Occlusion-Voting Space Carving": lambda: reconstruct_thermal_space_carving(
            dataset["thermal_images"],
            cameras,
            grid_res=grid_res,
            min_view_consistency=0.65,
        ),
        "2.  AOS Thermal Focal Stack (Schedl 2020)": lambda: reconstruct_aos_focal_stack(
            dataset["thermal_images"],
            cameras,
            grid_res=grid_res,
            trim_fraction=0.30,
        ),
        "3.  Physics-Robust Thermal Field (Ours)": lambda: reconstruct_thermal_radiance_field(
            dataset["thermal_images"],
            cameras,
            grid_res=grid_res,
            num_iters=35,
        ),
    }

    os.makedirs(export_dir, exist_ok=True)
    export_point_cloud_ply(os.path.join(export_dir, f"gt_{tag}.ply"), pts_gt)

    results = {}
    print(f"\n{'Method':<42} | {'IoU (↑)':>8} | {'Chamfer (↓)':>11} | {'F-Score@0.06 (↑)':>16}")
    print("-" * 86)
    for name, fn in methods.items():
        out = fn()
        iou = volumetric_iou(out["occupancy"], occ_gt)
        cd = chamfer_distance(out["surface_points"], pts_gt)
        f1 = f_score_3d(out["surface_points"], pts_gt, threshold=0.06)
        results[name] = {"iou": iou, "chamfer": cd, "f_score": f1, "num_pts": len(out["surface_points"])}
        print(f"{name:<42} | {iou:8.3f} | {cd:11.4f} | {f1:16.3f}")

        safe_name = name.split(".")[0].strip() + "_" + tag
        if len(out["surface_points"]) > 0:
            export_point_cloud_ply(
                os.path.join(export_dir, f"{safe_name}.ply"),
                out["surface_points"],
                out.get("surface_temperatures"),
            )

    return results


def run_nir_photometric_stereo_benchmark(export_dir: str = "outputs") -> dict:
    """Evaluate active Near-Infrared (NIR) Photometric Stereo under cast-shadow occlusions."""
    print(f"\n{'=' * 78}")
    print("Scenario: Active Near-Infrared (NIR) Photometric Stereo under Light Occlusions")
    print(f"{'=' * 78}")

    data = render_nir_photometric_stereo_dataset(grid_size=64, num_lights=8, occlude_lights=True)
    mask = data["mask"]
    Z_gt = data["Z_gt"]
    normals_gt = data["normals_gt"]

    # Baseline OLS (1 iteration, uniform weights) vs Robust IRLS (6 iterations)
    out_ols = reconstruct_ir_photometric_stereo(
        data["nir_images"], data["light_dirs"], mask, data["X"], data["Y"], irls_iters=1
    )
    out_irls = reconstruct_ir_photometric_stereo(
        data["nir_images"], data["light_dirs"], mask, data["X"], data["Y"], irls_iters=6
    )

    def eval_ps(out: dict) -> Tuple[float, float]:
        z_mae = float(np.mean(np.abs(out["Z_reconstructed"][mask] - Z_gt[mask])))
        cos_sim = np.clip(np.sum(out["normals_reconstructed"][mask] * normals_gt[mask], axis=-1), -1.0, 1.0)
        ang_err_deg = float(np.mean(np.rad2deg(np.arccos(cos_sim))))
        return z_mae, ang_err_deg

    mae_ols, ang_ols = eval_ps(out_ols)
    mae_irls, ang_irls = eval_ps(out_irls)

    print(f"{'Method':<42} | {'Height MAE (↓)':>14} | {'Normal Angular Error (↓)':>24}")
    print("-" * 86)
    print(f"{'Standard Least Squares (L2 Photometric)':<42} | {mae_ols:14.4f} | {ang_ols:21.2f} deg")
    print(f"{'Robust IRLS NIR Photometric Stereo':<42} | {mae_irls:14.4f} | {ang_irls:21.2f} deg")

    os.makedirs(export_dir, exist_ok=True)
    export_point_cloud_ply(
        os.path.join(export_dir, "nir_ps_irls.ply"),
        out_irls["surface_points"],
        out_irls["Z_reconstructed"][mask],
    )
    return {
        "ols": {"z_mae": mae_ols, "normal_err_deg": ang_ols},
        "irls": {"z_mae": mae_irls, "normal_err_deg": ang_irls},
    }


if __name__ == "__main__":
    res_clean = run_multiview_thermal_benchmark(num_occluders=0)
    res_occ = run_multiview_thermal_benchmark(num_occluders=6)
    res_nir = run_nir_photometric_stereo_benchmark()
    print(f"\nSaved reconstructed 3D point clouds (.ply) to ./outputs/")
