"""
Infrared 3D Object Reconstruction Testbed (with Occlusion Support).
"""

from .synthetic_data import (
    CameraPose,
    SyntheticIRScene,
    generate_turntable_cameras,
    render_multiview_ir_dataset,
    render_nir_photometric_stereo_dataset,
)
from .methods import (
    reconstruct_thermal_space_carving,
    reconstruct_aos_focal_stack,
    reconstruct_ir_photometric_stereo,
    reconstruct_thermal_radiance_field,
)
from .metrics import (
    chamfer_distance,
    f_score_3d,
    volumetric_iou,
    export_point_cloud_ply,
)

__all__ = [
    "CameraPose",
    "SyntheticIRScene",
    "generate_turntable_cameras",
    "render_multiview_ir_dataset",
    "render_nir_photometric_stereo_dataset",
    "reconstruct_thermal_space_carving",
    "reconstruct_aos_focal_stack",
    "reconstruct_ir_photometric_stereo",
    "reconstruct_thermal_radiance_field",
    "chamfer_distance",
    "f_score_3d",
    "volumetric_iou",
    "export_point_cloud_ply",
]

