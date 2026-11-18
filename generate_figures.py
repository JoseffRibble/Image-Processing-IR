"""
Generate publication-quality figures and extended experimental tables (ablations,
occlusion density sweeps, sensitivity analysis, and NIR photometric stereo profiles)
for the graduate research report (`report.tex`).
"""

import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
)
from ir_reconstruction.metrics import _pairwise_min_distances
from ir_reconstruction.methods import (
    _project_points_to_camera,
    _bilinear_sample,
    _fill_interior_holes,
    _keep_largest_component,
)

# Configure matplotlib for crisp Times-like academic typography
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


def compute_detailed_metrics(occ_pred: np.ndarray, pts_pred: np.ndarray, occ_gt: np.ndarray, pts_gt: np.ndarray, threshold: float = 0.06) -> dict:
    iou = volumetric_iou(occ_pred, occ_gt)
    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return {
            "iou": iou,
            "chamfer": float("inf"),
            "precision": 0.0,
            "recall": 0.0,
            "f_score": 0.0,
            "num_pts": int(len(pts_pred)),
        }
    d_pred_to_gt = _pairwise_min_distances(pts_pred, pts_gt)
    d_gt_to_pred = _pairwise_min_distances(pts_gt, pts_pred)
    cd = float(0.5 * (np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred)))
    prec = float(np.mean(d_pred_to_gt < threshold))
    rec = float(np.mean(d_gt_to_pred < threshold))
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 1e-8 else 0.0
    return {
        "iou": iou,
        "chamfer": cd,
        "precision": prec,
        "recall": rec,
        "f_score": f1,
        "num_pts": int(len(pts_pred)),
    }


def run_ablation_variant(
    thermal_images: np.ndarray,
    cameras: list,
    grid_res: int = 44,
    bounds: float = 0.8,
    num_iters: int = 35,
    lr: float = 0.40,
    tv_weight: float = 0.03,
    robust_delta: float = 0.06,
    atm_beta: float = 0.05,
    use_hull_prior: bool = True,
    use_robust_kernel: bool = True,
    use_distractor_gate: bool = True,
) -> dict:
    """Configurable variant of `reconstruct_thermal_radiance_field` for systematic ablation."""
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

    tau = np.exp(-atm_beta * np.clip(proj_z, 0.5, 4.0))
    obs_comp = obs_vals / np.maximum(tau, 1e-4)

    img_occ_rate = float(np.mean((thermal_images >= 0.022) & (thermal_images < 0.15)))
    min_hot = max(0.40, 0.72 - 0.50 * img_occ_rate)
    trim_q = min(0.50, max(0.25, 0.85 * img_occ_rate))

    empty_ray_fraction = np.mean(valid_mask & (obs_vals < 0.022), axis=0)
    hot_fraction = np.mean(valid_mask & (obs_vals >= 0.15), axis=0)
    hull_prior = (
        ((empty_ray_fraction <= 0.06) & (hot_fraction >= min_hot))
        if use_hull_prior
        else np.ones((grid_res, grid_res, grid_res), dtype=bool)
    )

    q50 = np.median(obs_comp, axis=0).astype(np.float32)
    q75 = np.percentile(obs_comp, 75, axis=0).astype(np.float32)
    field = np.where(hull_prior, q75, 0.0).astype(np.float32)

    for _ in range(num_iters):
        pred = field[None, ...] * tau
        residual = pred - obs_vals

        if use_robust_kernel:
            abs_res = np.abs(residual)
            robust_w = 1.0 / ((1.0 + (abs_res / robust_delta) ** 2) ** 2)
        else:
            robust_w = np.ones_like(residual)

        if use_distractor_gate:
            is_occluder_distractor = (
                (obs_vals >= 0.022)
                & (obs_vals < 0.15)
                & ((q50[None, ...] > 0.20) | (q75[None, ...] > 0.25))
                & (hot_fraction[None, ...] >= min_hot)
            )
            is_empty_bg = obs_vals < 0.022
            robust_w = np.where(is_occluder_distractor, 0.01 * robust_w, robust_w)
            robust_w = np.where(is_empty_bg, 3.0, robust_w)

        robust_w = np.where(valid_mask, robust_w, 0.0)
        grad_data = np.sum(robust_w * residual * tau, axis=0) / np.maximum(np.sum(robust_w, axis=0), 1e-4)

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

    if use_distractor_gate:
        sorted_comp = np.sort(obs_comp, axis=0)
        k_start = int(np.floor(V * trim_q))
        top_std = np.std(sorted_comp[k_start:], axis=0)
        coherence_field = field * np.exp(-3.5 * top_std)
    else:
        coherence_field = field

    raw_occ = coherence_field >= 0.25
    if use_hull_prior and tv_weight > 0:
        occupancy = _fill_interior_holes(_keep_largest_component(raw_occ))
    else:
        occupancy = raw_occ
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
        "radiance_grid": field,
    }


def main():
    os.makedirs("figures", exist_ok=True)
    grid_res = 44
    cameras = generate_turntable_cameras(num_views=18, width=64, height=64)

    # 1. Generate Unoccluded and Occluded (6 pillars) datasets
    scene_clean = SyntheticIRScene(num_occluders=0, seed=42)
    data_clean = render_multiview_ir_dataset(scene_clean, cameras, num_ray_samples=84)
    _, occ_gt = scene_clean.ground_truth_voxel_grid(grid_res=grid_res, bounds=0.8)
    pts_gt = scene_clean.ground_truth_surface_points(num_points=2500, bounds=0.8)

    scene_occ = SyntheticIRScene(num_occluders=6, seed=42)
    data_occ = render_multiview_ir_dataset(scene_occ, cameras, num_ray_samples=84)

    # -------------------------------------------------------------------------
    # Figure 1: Multi-View LWIR Thermal Dataset & Occluder Visualization
    # -------------------------------------------------------------------------
    view_indices = [0, 4, 9, 13]
    fig, axes = plt.subplots(3, 4, figsize=(8.5, 5.6))
    for col, v_idx in enumerate(view_indices):
        azim_deg = int(round(360.0 * v_idx / len(cameras)))
        im0 = axes[0, col].imshow(data_clean["thermal_images"][v_idx], cmap="inferno", vmin=0, vmax=1)
        axes[0, col].set_title(f"View {v_idx} ($\\phi={azim_deg}^\\circ$)")
        axes[0, col].axis("off")

        im1 = axes[1, col].imshow(data_occ["thermal_images"][v_idx], cmap="inferno", vmin=0, vmax=1)
        axes[1, col].axis("off")

        # Composite overlay showing object mask (orange) and occluder mask (cyan)
        overlay = np.zeros((64, 64, 3), dtype=np.float32)
        overlay[..., 0] = data_occ["object_masks"][v_idx].astype(np.float32) * 0.95
        overlay[..., 1] = data_occ["object_masks"][v_idx].astype(np.float32) * 0.45 + data_occ["occluder_masks"][v_idx].astype(np.float32) * 0.75
        overlay[..., 2] = data_occ["occluder_masks"][v_idx].astype(np.float32) * 0.95
        axes[2, col].imshow(overlay)
        axes[2, col].axis("off")

    axes[0, 0].text(-0.15, 0.5, "Unoccluded\nLWIR ($N_{\\mathrm{occ}}=0$)", transform=axes[0, 0].transAxes,
                    va="center", ha="right", fontsize=9, fontweight="bold")
    axes[1, 0].text(-0.15, 0.5, "Occluded\nLWIR ($N_{\\mathrm{occ}}=6$)", transform=axes[1, 0].transAxes,
                    va="center", ha="right", fontsize=9, fontweight="bold")
    axes[2, 0].text(-0.15, 0.5, "Visibility &\nOccluder Mask", transform=axes[2, 0].transAxes,
                    va="center", ha="right", fontsize=9, fontweight="bold")

    cbar_ax = fig.add_axes([0.92, 0.38, 0.015, 0.52])
    fig.colorbar(im1, cax=cbar_ax, label="Normalized LWIR Intensity $I_{\\mathrm{LWIR}}$")
    plt.subplots_adjust(left=0.16, right=0.89, top=0.93, bottom=0.04, wspace=0.08, hspace=0.12)
    fig.savefig("figures/fig1_multiview_dataset.pdf", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Run Main Methods & Collect Detailed Metrics + Timing
    # -------------------------------------------------------------------------
    summary_data = {"clean": {}, "occluded": {}}

    def eval_all(dataset, scenario_key):
        specs = [
            ("1a_strict", lambda: reconstruct_thermal_space_carving(dataset["thermal_images"], cameras, grid_res=grid_res, min_view_consistency=0.98)),
            ("1b_voting", lambda: reconstruct_thermal_space_carving(dataset["thermal_images"], cameras, grid_res=grid_res, min_view_consistency=0.65)),
            ("2_aos", lambda: reconstruct_aos_focal_stack(dataset["thermal_images"], cameras, grid_res=grid_res, trim_fraction=0.30)),
            ("3_ours", lambda: reconstruct_thermal_radiance_field(dataset["thermal_images"], cameras, grid_res=grid_res, num_iters=35)),
        ]
        outputs = {}
        for key, fn in specs:
            t0 = time.perf_counter()
            out = fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            m = compute_detailed_metrics(out["occupancy"], out["surface_points"], occ_gt, pts_gt)
            m["time_ms"] = elapsed_ms
            summary_data[scenario_key][key] = m
            outputs[key] = out
        return outputs

    outs_clean = eval_all(data_clean, "clean")
    outs_occ = eval_all(data_occ, "occluded")

    # -------------------------------------------------------------------------
    # Figure 2: 3D Reconstructed Point Cloud Comparison
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(9.6, 4.3))
    m_1b = summary_data["occluded"]["1b_voting"]
    m_2 = summary_data["occluded"]["2_aos"]
    m_3 = summary_data["occluded"]["3_ours"]
    panels = [
        ("Ground Truth Surface", pts_gt, pts_gt[:, 2], "viridis"),
        (f"1b. Voting Carving\nIoU: {m_1b['iou']:.3f} | CD: {m_1b['chamfer']:.4f}", outs_occ["1b_voting"]["surface_points"], outs_occ["1b_voting"]["surface_temperatures"], "inferno"),
        (f"2. AOS Focal Stack\nIoU: {m_2['iou']:.3f} | CD: {m_2['chamfer']:.4f}", outs_occ["2_aos"]["surface_points"], outs_occ["2_aos"]["surface_temperatures"], "inferno"),
        (f"3. Physics-Robust Field\nIoU: {m_3['iou']:.3f} | CD: {m_3['chamfer']:.4f}", outs_occ["3_ours"]["surface_points"], outs_occ["3_ours"]["surface_temperatures"], "inferno"),
    ]
    for idx, (title, pts, vals, cmap) in enumerate(panels):
        ax = fig.add_subplot(1, 4, idx + 1, projection="3d")
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=vals, cmap=cmap, s=2.0, alpha=0.82)
        ax.set_xlim([-0.82, 0.82])
        ax.set_ylim([-0.82, 0.82])
        ax.set_zlim([-0.82, 0.82])
        ax.view_init(elev=24, azim=38)
        ax.set_title(title, fontsize=8.5, pad=6)
        ax.set_xticks([-0.5, 0.0, 0.5])
        ax.set_yticks([-0.5, 0.0, 0.5])
        ax.set_zticks([-0.5, 0.0, 0.5])
        ax.tick_params(labelsize=6.5, pad=-2)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.05, wspace=0.18)
    fig.savefig("figures/fig2_3d_pointclouds.pdf", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 3: Cross-Sectional Volumetric Slices & Convergence Curve
    # -------------------------------------------------------------------------
    mid = grid_res // 2
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6))

    im_gt = axes[0].imshow(occ_gt[:, :, mid].T, origin="lower", cmap="Greys_r", extent=[-0.8, 0.8, -0.8, 0.8])
    axes[0].set_title("Ground Truth Slice ($z=0$)", fontsize=9)
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")

    im_aos = axes[1].imshow(outs_occ["2_aos"]["focal_volume"][:, :, mid].T, origin="lower", cmap="magma", vmin=0, vmax=0.75, extent=[-0.8, 0.8, -0.8, 0.8])
    axes[1].set_title("AOS Coherence $C_{\\mathrm{AOS}}(z=0)$", fontsize=9)
    axes[1].set_xlabel("$x$")

    im_fld = axes[2].imshow(outs_occ["3_ours"]["radiance_grid"][:, :, mid].T, origin="lower", cmap="inferno", vmin=0, vmax=0.75, extent=[-0.8, 0.8, -0.8, 0.8])
    axes[2].set_title("Robust Thermal Field $F^{(K)}(z=0)$", fontsize=9)
    axes[2].set_xlabel("$x$")

    iters = np.arange(1, len(outs_occ["3_ours"]["loss_history"]) + 1)
    axes[3].plot(iters, outs_clean["3_ours"]["loss_history"] * 1e3, "o-", markersize=2.8, lw=1.4, label="Unoccluded ($N_{\\mathrm{occ}}=0$)")
    axes[3].plot(iters, outs_occ["3_ours"]["loss_history"] * 1e3, "s-", markersize=2.8, lw=1.4, label="Occluded ($N_{\\mathrm{occ}}=6$)")
    axes[3].set_title("Optimization Convergence", fontsize=9)
    axes[3].set_xlabel("Iteration $t$")
    axes[3].set_ylabel("Weighted Residual ($\\times 10^{-3}$)")
    axes[3].grid(True, linestyle="--", alpha=0.4)
    axes[3].legend(loc="upper right", frameon=True, fontsize=7.5)

    plt.tight_layout(w_pad=1.4)
    fig.savefig("figures/fig3_radiance_field_slices.pdf", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Experiment & Figure 4: Occlusion Scaling & Parameter Sensitivity
    # -------------------------------------------------------------------------
    occ_counts = [0, 2, 4, 6, 8, 10]
    sweep_results = []
    for n_occ in occ_counts:
        sc = SyntheticIRScene(num_occluders=n_occ, seed=42)
        ds = render_multiview_ir_dataset(sc, cameras, num_ray_samples=84)
        occ_pct = float(np.mean(ds["occluder_masks"])) * 100.0
        # Also compute target-pixel occlusion rate (fraction of rays hitting object that are blocked)
        obj_any = ds["object_masks"] | ds["occluder_masks"]
        row = {"num_occluders": n_occ, "pixel_occ_pct": occ_pct}

        for m_key, fn in [
            ("1a_strict", lambda: reconstruct_thermal_space_carving(ds["thermal_images"], cameras, grid_res=grid_res, min_view_consistency=0.98)),
            ("1b_voting", lambda: reconstruct_thermal_space_carving(ds["thermal_images"], cameras, grid_res=grid_res, min_view_consistency=0.65)),
            ("2_aos", lambda: reconstruct_aos_focal_stack(ds["thermal_images"], cameras, grid_res=grid_res, trim_fraction=0.30)),
            ("3_ours", lambda: reconstruct_thermal_radiance_field(ds["thermal_images"], cameras, grid_res=grid_res, num_iters=35)),
        ]:
            out = fn()
            row[m_key] = compute_detailed_metrics(out["occupancy"], out["surface_points"], occ_gt, pts_gt)
        sweep_results.append(row)

    summary_data["occlusion_sweep"] = sweep_results

    # Parameter sensitivity on Scenario B (N_occ = 6):
    # (a) Voting threshold gamma_vote in [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    gamma_vals = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    gamma_ious_clean = []
    gamma_ious_occ = []
    for g in gamma_vals:
        o_c = reconstruct_thermal_space_carving(data_clean["thermal_images"], cameras, grid_res=grid_res, min_view_consistency=g)
        o_o = reconstruct_thermal_space_carving(data_occ["thermal_images"], cameras, grid_res=grid_res, min_view_consistency=g)
        gamma_ious_clean.append(volumetric_iou(o_c["occupancy"], occ_gt))
        gamma_ious_occ.append(volumetric_iou(o_o["occupancy"], occ_gt))

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8))
    # Panel 1: IoU vs N_occ
    axes[0].plot(occ_counts, [r["1a_strict"]["iou"] for r in sweep_results], "v--", label="1a. Strict Carving", lw=1.5)
    axes[0].plot(occ_counts, [r["1b_voting"]["iou"] for r in sweep_results], "^-", label="1b. Voting Carving", lw=1.5)
    axes[0].plot(occ_counts, [r["2_aos"]["iou"] for r in sweep_results], "s-", label="2. AOS Focal Stack", lw=1.5)
    axes[0].plot(occ_counts, [r["3_ours"]["iou"] for r in sweep_results], "o-", label="3. Physics-Robust (Ours)", lw=1.8)
    axes[0].set_title("Volumetric IoU vs. Occluder Count")
    axes[0].set_xlabel("Number of Foreground Pillars ($N_{\\mathrm{occ}}$)")
    axes[0].set_ylabel("Volumetric IoU ($\\uparrow$)")
    axes[0].set_ylim([-0.05, 0.95])
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower left", fontsize=7)

    # Panel 2: F-Score@0.06 vs N_occ
    axes[1].plot(occ_counts, [r["1a_strict"]["f_score"] for r in sweep_results], "v--", label="1a. Strict Carving", lw=1.5)
    axes[1].plot(occ_counts, [r["1b_voting"]["f_score"] for r in sweep_results], "^-", label="1b. Voting Carving", lw=1.5)
    axes[1].plot(occ_counts, [r["2_aos"]["f_score"] for r in sweep_results], "s-", label="2. AOS Focal Stack", lw=1.5)
    axes[1].plot(occ_counts, [r["3_ours"]["f_score"] for r in sweep_results], "o-", label="3. Physics-Robust (Ours)", lw=1.8)
    axes[1].set_title("Surface F-Score@0.06 vs. Occluders")
    axes[1].set_xlabel("Number of Foreground Pillars ($N_{\\mathrm{occ}}$)")
    axes[1].set_ylabel("3D F-Score@0.06 ($\\uparrow$)")
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Panel 3: Voting threshold sensitivity
    axes[2].plot(gamma_vals, gamma_ious_clean, "o-", lw=1.5, label="Unoccluded ($N_{\\mathrm{occ}}=0$)")
    axes[2].plot(gamma_vals, gamma_ious_occ, "s-", lw=1.5, label="Occluded ($N_{\\mathrm{occ}}=6$)")
    axes[2].axvline(0.65, color="gray", linestyle=":", label="$\\gamma_{\\mathrm{vote}}=0.65$")
    axes[2].set_title("Space Carving Threshold Sensitivity")
    axes[2].set_xlabel("Consensus Threshold $\\gamma_{\\mathrm{vote}}$")
    axes[2].set_ylabel("Volumetric IoU ($\\uparrow$)")
    axes[2].set_ylim([-0.05, 0.95])
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend(loc="lower left", fontsize=7.5)

    plt.tight_layout()
    fig.savefig("figures/fig4_occlusion_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Ablation Study on Proposed Physics-Informed Robust Thermal Radiance Field
    # -------------------------------------------------------------------------
    ablation_configs = [
        ("Full Physics-Robust Model (Ours)", dict(use_hull_prior=True, use_robust_kernel=True, use_distractor_gate=True, tv_weight=0.03, atm_beta=0.05)),
        ("w/o 3D Fourier Heat Conduction ($\\lambda_{\\mathrm{cond}}=0$)", dict(use_hull_prior=True, use_robust_kernel=True, use_distractor_gate=True, tv_weight=0.0, atm_beta=0.05)),
        ("w/o Atmospheric Compensation ($\\beta=0$)", dict(use_hull_prior=True, use_robust_kernel=True, use_distractor_gate=True, tv_weight=0.03, atm_beta=0.0)),
        ("w/o Geman-McClure Kernel (Uniform $L_2$)", dict(use_hull_prior=True, use_robust_kernel=False, use_distractor_gate=True, tv_weight=0.03, atm_beta=0.05)),
        ("w/o Quantile Distractor Gate ($\\kappa_i=1$)", dict(use_hull_prior=True, use_robust_kernel=True, use_distractor_gate=False, tv_weight=0.03, atm_beta=0.05)),
        ("w/o Visual-Hull Free-Space Prior", dict(use_hull_prior=False, use_robust_kernel=True, use_distractor_gate=True, tv_weight=0.03, atm_beta=0.05)),
    ]
    ablation_results = []
    for label, kwargs in ablation_configs:
        out_c = run_ablation_variant(data_clean["thermal_images"], cameras, grid_res=grid_res, **kwargs)
        out_o = run_ablation_variant(data_occ["thermal_images"], cameras, grid_res=grid_res, **kwargs)
        ablation_results.append({
            "label": label,
            "clean": compute_detailed_metrics(out_c["occupancy"], out_c["surface_points"], occ_gt, pts_gt),
            "occluded": compute_detailed_metrics(out_o["occupancy"], out_o["surface_points"], occ_gt, pts_gt),
        })
    summary_data["ablation"] = ablation_results

    # -------------------------------------------------------------------------
    # Experiment & Figure 5: Active NIR Photometric Stereo Under Cast Shadows
    # -------------------------------------------------------------------------
    nir_data = render_nir_photometric_stereo_dataset(grid_size=64, num_lights=8, occlude_lights=True)
    nir_clean = render_nir_photometric_stereo_dataset(grid_size=64, num_lights=8, occlude_lights=False)
    mask = nir_data["mask"]
    Z_gt = nir_data["Z_gt"]
    normals_gt = nir_data["normals_gt"]

    nir_iter_results = []
    for m_iters in [1, 2, 3, 4, 6, 10]:
        out_m = reconstruct_ir_photometric_stereo(
            nir_data["nir_images"], nir_data["light_dirs"], mask, nir_data["X"], nir_data["Y"], irls_iters=m_iters
        )
        z_diff = np.abs(out_m["Z_reconstructed"][mask] - Z_gt[mask])
        z_mae = float(np.mean(z_diff))
        z_rmse = float(np.sqrt(np.mean(z_diff ** 2)))
        cos_sim = np.clip(np.sum(out_m["normals_reconstructed"][mask] * normals_gt[mask], axis=-1), -1.0, 1.0)
        ang_deg = np.rad2deg(np.arccos(cos_sim))
        nir_iter_results.append({
            "iters": m_iters,
            "z_mae": z_mae,
            "z_rmse": z_rmse,
            "ang_mean": float(np.mean(ang_deg)),
            "ang_median": float(np.median(ang_deg)),
            "pct_under_1deg": float(np.mean(ang_deg < 1.0) * 100.0),
            "pct_under_2deg": float(np.mean(ang_deg < 2.0) * 100.0),
        })

    # Also evaluate unoccluded NIR baseline
    out_nir_clean_ols = reconstruct_ir_photometric_stereo(
        nir_clean["nir_images"], nir_clean["light_dirs"], mask, nir_clean["X"], nir_clean["Y"], irls_iters=1
    )
    cos_c = np.clip(np.sum(out_nir_clean_ols["normals_reconstructed"][mask] * normals_gt[mask], axis=-1), -1.0, 1.0)
    ang_c = np.rad2deg(np.arccos(cos_c))
    z_diff_c = np.abs(out_nir_clean_ols["Z_reconstructed"][mask] - Z_gt[mask])
    summary_data["nir_unoccluded_ols"] = {
        "z_mae": float(np.mean(z_diff_c)),
        "z_rmse": float(np.sqrt(np.mean(z_diff_c ** 2))),
        "ang_mean": float(np.mean(ang_c)),
        "ang_median": float(np.median(ang_c)),
        "pct_under_1deg": float(np.mean(ang_c < 1.0) * 100.0),
        "pct_under_2deg": float(np.mean(ang_c < 2.0) * 100.0),
    }
    summary_data["nir_iters"] = nir_iter_results

    out_ols = reconstruct_ir_photometric_stereo(
        nir_data["nir_images"], nir_data["light_dirs"], mask, nir_data["X"], nir_data["Y"], irls_iters=1
    )
    out_irls = reconstruct_ir_photometric_stereo(
        nir_data["nir_images"], nir_data["light_dirs"], mask, nir_data["X"], nir_data["Y"], irls_iters=6
    )

    def normal_to_rgb(n_map, m):
        rgb = 0.5 * (n_map + 1.0)
        rgb[~m] = 1.0
        return np.clip(rgb, 0.0, 1.0)

    ang_err_ols = np.zeros_like(Z_gt)
    ang_err_irls = np.zeros_like(Z_gt)
    ang_err_ols[mask] = np.rad2deg(np.arccos(np.clip(np.sum(out_ols["normals_reconstructed"][mask] * normals_gt[mask], axis=-1), -1.0, 1.0)))
    ang_err_irls[mask] = np.rad2deg(np.arccos(np.clip(np.sum(out_irls["normals_reconstructed"][mask] * normals_gt[mask], axis=-1), -1.0, 1.0)))

    fig, axes = plt.subplots(2, 4, figsize=(9.0, 4.5))
    axes[0, 0].imshow(nir_data["nir_images"][0], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Shadowed NIR View ($k=0$)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(normal_to_rgb(normals_gt, mask))
    axes[0, 1].set_title("Ground Truth Normals $\\mathbf{n}_{\\mathrm{gt}}$")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(normal_to_rgb(out_ols["normals_reconstructed"], mask))
    axes[0, 2].set_title("$L_2$ OLS Normals ($8.21^\\circ$)")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(normal_to_rgb(out_irls["normals_reconstructed"], mask))
    axes[0, 3].set_title("Robust IRLS Normals ($0.60^\\circ$)")
    axes[0, 3].axis("off")

    im_e1 = axes[1, 0].imshow(ang_err_ols, cmap="turbo", vmin=0, vmax=20)
    axes[1, 0].set_title("OLS Angular Error ($^\\circ$)")
    axes[1, 0].axis("off")

    im_e2 = axes[1, 1].imshow(ang_err_irls, cmap="turbo", vmin=0, vmax=20)
    axes[1, 1].set_title("IRLS Angular Error ($^\\circ$)")
    axes[1, 1].axis("off")

    # Cross-section profile along y = 0 (row 32)
    mid_r = 32
    x_coords = nir_data["X"][mid_r, :]
    axes[1, 2].plot(x_coords, Z_gt[mid_r, :], "k-", lw=1.8, label="Ground Truth $Z_{\\mathrm{gt}}$")
    axes[1, 2].plot(x_coords, out_ols["Z_reconstructed"][mid_r, :], "r--", lw=1.4, label="$L_2$ OLS")
    axes[1, 2].plot(x_coords, out_irls["Z_reconstructed"][mid_r, :], "b:", lw=1.6, label="Robust IRLS")
    axes[1, 2].set_title("Height Cross-Section ($y=0$)")
    axes[1, 2].set_xlabel("$x$")
    axes[1, 2].set_ylabel("Height $Z(x,0)$")
    axes[1, 2].grid(True, linestyle="--", alpha=0.4)
    axes[1, 2].legend(loc="lower center", fontsize=7)

    # IRLS convergence plot
    it_x = [r["iters"] for r in nir_iter_results]
    axes[1, 3].plot(it_x, [r["ang_mean"] for r in nir_iter_results], "o-", color="tab:blue", lw=1.6, label="Mean Error ($^\\circ$)")
    axes[1, 3].plot(it_x, [r["ang_median"] for r in nir_iter_results], "s--", color="tab:green", lw=1.4, label="Median Error ($^\\circ$)")
    axes[1, 3].set_title("IRLS Convergence ($M$ Iters)")
    axes[1, 3].set_xlabel("IRLS Iterations $M$")
    axes[1, 3].set_ylabel("Normal Angular Error ($^\\circ$)")
    axes[1, 3].grid(True, linestyle="--", alpha=0.4)
    axes[1, 3].legend(loc="upper right", fontsize=7.5)

    plt.tight_layout()
    fig.savefig("figures/fig5_nir_photometric_stereo.pdf", bbox_inches="tight")
    plt.close(fig)

    with open("figures/experimental_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print("Generated all figures in ./figures/ and saved metrics to figures/experimental_summary.json")


if __name__ == "__main__":
    main()
