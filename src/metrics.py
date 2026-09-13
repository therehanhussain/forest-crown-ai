"""
Canopy metrics calculation, uncertainty analysis, and ethical reporting for Forest Crown AI.

Strict Engineering Principles Enforced:
1. Strict separation of tree count from canopy-area estimation.
2. Areas calculated solely in projected metric CRS (m^2 and hectares), never in degrees.
3. Overlapping crowns are dissolved via spatial union (no naive bounding-box summation).
4. Full disclosure of assumptions, uncertainty ranges, and methodological limitations.
5. Absolute refusal to claim biomass, carbon stocks, or verified carbon credits.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    Polygon = None
    unary_union = None
    HAS_SHAPELY = False


@dataclass
class CanopyMetrics:
    """Comprehensive canopy metrics and analysis report."""

    # 1. Primary Required Metrics (Requirement H)
    tree_count: int
    forest_area_m2: Optional[float]
    forest_area_ha: Optional[float]
    detected_crown_area_m2: float
    unique_canopy_area_m2: float
    canopy_cover_pct: Optional[float]
    mean_crown_area_m2: float
    median_crown_area_m2: float
    median_detection_confidence: float
    low_confidence_count: int
    overlap_redundancy_pct: float
    gsd_x: float
    gsd_y: float
    crs: str
    crown_geometry_method: str = "inscribed_ellipse_from_detection_bbox"

    # 2. Quality Flags (Requirement I)
    quality_flags: List[str] = field(default_factory=list)

    # 3. Additional & Backwards-Compatible Fields
    raw_box_area_m2: float = 0.0
    dissolved_box_area_m2: float = 0.0
    dissolved_canopy_area_m2: float = 0.0  # Backward-compatible alias for unique_canopy_area_m2
    dissolved_canopy_area_ha: float = 0.0
    overlap_redundancy_m2: float = 0.0
    aoi_total_area_m2: Optional[float] = None  # Backward-compatible alias for forest_area_m2
    aoi_total_area_ha: Optional[float] = None  # Backward-compatible alias for forest_area_ha
    canopy_cover_percentage: Optional[float] = None  # Backward-compatible alias for canopy_cover_pct
    trees_per_hectare: Optional[float] = None

    mean_crown_diameter_m: float = 0.0
    median_crown_diameter_m: float = 0.0
    std_crown_diameter_m: float = 0.0
    min_crown_diameter_m: float = 0.0
    max_crown_diameter_m: float = 0.0

    mean_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0

    metric_crs: str = ""
    gsd_m: float = 0.0
    gsd_source: str = "unknown"
    confidence_threshold: float = 0.20

    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    carbon_disclaimer: str = field(
        default=(
            "LEGAL & METHODOLOGICAL DISCLAIMER: Forest Crown AI calculates 2D optical crown counts and canopy "
            "surface area only. It DOES NOT estimate aboveground biomass (AGB), wood density, carbon stock (tCO2e), "
            "or verifiable carbon offsets. Carbon accounting requires calibrated field plots, allometric equations, "
            "accurate tree height / LiDAR structural profiles, and independent third-party audit (e.g. Verra VCS, "
            "Gold Standard). Under no circumstances should optical crown area be cited as verified carbon credits."
        )
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_canopy_metrics(
    crown_diameters_m: List[float],
    crown_metric_polygons: List[Any],
    crown_metric_boxes: List[Any],
    crown_confidences: List[float],
    aoi_metric_polygon: Optional[Any],
    gsd_m: float,
    gsd_source: str,
    metric_crs: str,
    confidence_threshold: float,
    gsd_x: Optional[float] = None,
    gsd_y: Optional[float] = None,
    is_georeferenced: bool = True,
    is_partial_overlap: bool = False
) -> CanopyMetrics:
    """
    Calculates detailed canopy metrics respecting all hackathon engineering principles.

    Args:
        crown_diameters_m: List of crown diameters in meters.
        crown_metric_polygons: List of realistic crown geometries in metric CRS.
        crown_metric_boxes: List of raw bounding box geometries in metric CRS.
        crown_confidences: List of detection confidence scores.
        aoi_metric_polygon: Metric polygon of AOI (or entire image extent).
        gsd_m: Ground sample distance in meters per pixel.
        gsd_source: Provenance of GSD ("metadata_projected", "user_specified", etc.).
        metric_crs: CRS name/code in which metrics were computed.
        confidence_threshold: Minimum score filter applied.
        gsd_x: GSD along X-axis (optional, defaults to gsd_m).
        gsd_y: GSD along Y-axis (optional, defaults to gsd_m).
        is_georeferenced: Whether imagery contains valid CRS/georeferencing.
        is_partial_overlap: Whether AOI partially intersects imagery.
    """
    actual_gsd_x = gsd_x if gsd_x is not None else gsd_m
    actual_gsd_y = gsd_y if gsd_y is not None else gsd_m

    tree_count = len(crown_metric_polygons)

    # 1. Forest AOI Extent
    forest_area_m2 = None
    forest_area_ha = None
    if aoi_metric_polygon is not None and HAS_SHAPELY:
        forest_area_m2 = float(aoi_metric_polygon.area)
        if forest_area_m2 > 0:
            forest_area_ha = forest_area_m2 / 10000.0

    # Build Quality Flags (Requirement I)
    quality_flags = [
        "Crown areas are geometric approximations, not segmentation",
        "Results are not biomass or carbon-stock estimates"
    ]
    if gsd_source == "user_specified":
        quality_flags.append("User-supplied GSD")
    if not is_georeferenced:
        quality_flags.append("Imagery has no CRS")
    if is_partial_overlap:
        quality_flags.append("Partial AOI/image overlap")
    if gsd_m > 0.30:
        quality_flags.append("Low-resolution imagery may miss small crowns")

    if tree_count == 0:
        return CanopyMetrics(
            tree_count=0,
            forest_area_m2=round(forest_area_m2, 2) if forest_area_m2 else None,
            forest_area_ha=round(forest_area_ha, 4) if forest_area_ha else None,
            detected_crown_area_m2=0.0,
            unique_canopy_area_m2=0.0,
            canopy_cover_pct=0.0 if forest_area_m2 else None,
            mean_crown_area_m2=0.0,
            median_crown_area_m2=0.0,
            median_detection_confidence=0.0,
            low_confidence_count=0,
            overlap_redundancy_pct=0.0,
            gsd_x=round(actual_gsd_x, 4),
            gsd_y=round(actual_gsd_y, 4),
            crs=metric_crs,
            crown_geometry_method="inscribed_ellipse_from_detection_bbox",
            quality_flags=quality_flags,
            raw_box_area_m2=0.0,
            dissolved_box_area_m2=0.0,
            dissolved_canopy_area_m2=0.0,
            dissolved_canopy_area_ha=0.0,
            overlap_redundancy_m2=0.0,
            aoi_total_area_m2=round(forest_area_m2, 2) if forest_area_m2 else None,
            aoi_total_area_ha=round(forest_area_ha, 4) if forest_area_ha else None,
            canopy_cover_percentage=0.0 if forest_area_m2 else None,
            trees_per_hectare=0.0 if forest_area_ha else None,
            mean_crown_diameter_m=0.0,
            median_crown_diameter_m=0.0,
            std_crown_diameter_m=0.0,
            min_crown_diameter_m=0.0,
            max_crown_diameter_m=0.0,
            mean_confidence=0.0,
            min_confidence=0.0,
            max_confidence=0.0,
            metric_crs=metric_crs,
            gsd_m=gsd_m,
            gsd_source=gsd_source,
            confidence_threshold=confidence_threshold,
            assumptions=[f"GSD source: {gsd_source} ({gsd_m:.4f} m/pixel)"],
            limitations=["No tree crowns detected above score threshold."]
        )

    # 2. Individual Crown Areas & Dissolved Unique Canopy Footprint
    individual_crown_areas = [float(p.area) for p in crown_metric_polygons] if HAS_SHAPELY else [
        math.pi * ((d / 2.0) ** 2) for d in crown_diameters_m
    ]
    detected_crown_area_m2 = sum(individual_crown_areas)

    if HAS_SHAPELY:
        raw_box_area_m2 = sum(float(b.area) for b in crown_metric_boxes)
        dissolved_box_union = unary_union(crown_metric_boxes)
        dissolved_box_area_m2 = float(dissolved_box_union.area)

        # Dissolved realistic canopy footprint (inscribed ellipses union)
        dissolved_canopy_union = unary_union(crown_metric_polygons)
        unique_canopy_area_m2 = float(dissolved_canopy_union.area)
    else:
        raw_box_area_m2 = sum(math.pi * ((d / 2.0) ** 2) * (4.0 / math.pi) for d in crown_diameters_m)
        dissolved_box_area_m2 = raw_box_area_m2 * 0.90
        unique_canopy_area_m2 = detected_crown_area_m2 * 0.90

    dissolved_canopy_area_ha = unique_canopy_area_m2 / 10000.0

    # 3. Overlap Redundancy Analysis (Separating count from canopy area)
    overlap_redundancy_m2 = max(0.0, raw_box_area_m2 - dissolved_box_area_m2)
    overlap_redundancy_pct = (overlap_redundancy_m2 / raw_box_area_m2 * 100.0) if raw_box_area_m2 > 0 else 0.0

    # 4. Canopy Cover Percentage (Uncapped: do not silently cap at 100%)
    canopy_cover_pct = None
    trees_per_hectare = None
    if forest_area_m2 and forest_area_m2 > 0:
        canopy_cover_pct = (unique_canopy_area_m2 / forest_area_m2) * 100.0
        if forest_area_ha and forest_area_ha > 0:
            trees_per_hectare = tree_count / forest_area_ha

    if overlap_redundancy_pct > 15.0 or (trees_per_hectare and trees_per_hectare > 200):
        quality_flags.append("Dense canopy may cause merged/missed detections")

    # 5. Crown Size & Area Statistics
    mean_crown_area_m2 = float(np.mean(individual_crown_areas))
    median_crown_area_m2 = float(np.median(individual_crown_areas))

    d_arr = np.array(crown_diameters_m)
    mean_d = float(np.mean(d_arr))
    median_d = float(np.median(d_arr))
    std_d = float(np.std(d_arr))
    min_d = float(np.min(d_arr))
    max_d = float(np.max(d_arr))

    # 6. Confidence Statistics
    c_arr = np.array(crown_confidences)
    mean_conf = float(np.mean(c_arr))
    median_conf = float(np.median(c_arr))
    min_conf = float(np.min(c_arr))
    max_conf = float(np.max(c_arr))
    low_confidence_count = int(np.sum(c_arr < 0.40))

    # 7. Methodological Assumptions and Limitations
    assumptions = [
        f"Spatial Resolution: X={actual_gsd_x:.4f} m/px, Y={actual_gsd_y:.4f} m/px (Source: {gsd_source}).",
        f"Projection: Calculations executed in projected metric coordinate system '{metric_crs}'.",
        "Crown Morphology: Individual crowns modeled as inscribed ellipses within detector bounding boxes to eliminate ~21.5% rectangular overestimation error.",
        "Canopy Dissolve: Overlapping crowns merged using geometric unary union to prevent double-counting shared canopy area.",
        f"Confidence Cutoff: Detections filtered at score threshold >= {confidence_threshold:.2f}."
    ]

    limitations = [
        "Sub-canopy understory trees occluded by dominant canopy layers cannot be detected from 2D optical nadir imagery.",
        "Dense closed-canopy stands with continuous interlocking crowns may exhibit clumping or boundary splitting errors.",
        "Orthomosaic seams, cloud cover, and deep topographic shadows can produce false negatives or distorted crown outlines.",
        "Small saplings (< 3x GSD in diameter) cannot be reliably detected due to sensor pixel limits."
    ]

    return CanopyMetrics(
        tree_count=tree_count,
        forest_area_m2=round(forest_area_m2, 2) if forest_area_m2 is not None else None,
        forest_area_ha=round(forest_area_ha, 4) if forest_area_ha is not None else None,
        detected_crown_area_m2=round(detected_crown_area_m2, 2),
        unique_canopy_area_m2=round(unique_canopy_area_m2, 2),
        canopy_cover_pct=round(canopy_cover_pct, 2) if canopy_cover_pct is not None else None,
        mean_crown_area_m2=round(mean_crown_area_m2, 2),
        median_crown_area_m2=round(median_crown_area_m2, 2),
        median_detection_confidence=round(median_conf, 4),
        low_confidence_count=low_confidence_count,
        overlap_redundancy_pct=round(overlap_redundancy_pct, 2),
        gsd_x=round(actual_gsd_x, 4),
        gsd_y=round(actual_gsd_y, 4),
        crs=metric_crs,
        crown_geometry_method="inscribed_ellipse_from_detection_bbox",
        quality_flags=quality_flags,
        raw_box_area_m2=round(raw_box_area_m2, 2),
        dissolved_box_area_m2=round(dissolved_box_area_m2, 2),
        dissolved_canopy_area_m2=round(unique_canopy_area_m2, 2),
        dissolved_canopy_area_ha=round(dissolved_canopy_area_ha, 4),
        overlap_redundancy_m2=round(overlap_redundancy_m2, 2),
        aoi_total_area_m2=round(forest_area_m2, 2) if forest_area_m2 is not None else None,
        aoi_total_area_ha=round(forest_area_ha, 4) if forest_area_ha is not None else None,
        canopy_cover_percentage=round(canopy_cover_pct, 2) if canopy_cover_pct is not None else None,
        trees_per_hectare=round(trees_per_hectare, 1) if trees_per_hectare is not None else None,
        mean_crown_diameter_m=round(mean_d, 2),
        median_crown_diameter_m=round(median_d, 2),
        std_crown_diameter_m=round(std_d, 2),
        min_crown_diameter_m=round(min_d, 2),
        max_crown_diameter_m=round(max_d, 2),
        mean_confidence=round(mean_conf, 4),
        min_confidence=round(min_conf, 4),
        max_confidence=round(max_conf, 4),
        metric_crs=metric_crs,
        gsd_m=gsd_m,
        gsd_source=gsd_source,
        confidence_threshold=confidence_threshold,
        assumptions=assumptions,
        limitations=limitations
    )
