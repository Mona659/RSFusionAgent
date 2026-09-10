"""Executable tools exposed to the future RSFusionAgent workflow."""

from rsfusion_agent.tools.raster_inspector import RasterInspectionResult, inspect_raster
from rsfusion_agent.tools.raster_preview import render_raster_rgb
from rsfusion_agent.tools.tiff_crop import TiffCropResult, crop_tiff_triplet, load_crop_manifest
from rsfusion_agent.tools.tiff_patch import (
    PreparedTiffPatch,
    inspect_manifest_crop_triplet,
    prepare_tiff_patch,
    prepare_tiff_patch_from_manifest,
)
from rsfusion_agent.tools.tiff_triplet import TiffTripletInspection, inspect_tiff_triplet

__all__ = [
    "RasterInspectionResult",
    "TiffCropResult",
    "PreparedTiffPatch",
    "TiffTripletInspection",
    "crop_tiff_triplet",
    "inspect_manifest_crop_triplet",
    "inspect_raster",
    "render_raster_rgb",
    "inspect_tiff_triplet",
    "load_crop_manifest",
    "prepare_tiff_patch",
    "prepare_tiff_patch_from_manifest",
]
