"""
End-to-End Forest Crown AI Processing Pipeline.

Orchestrates:
1. Input validation & strict GSD provenance verification
2. KML/KMZ boundary loading & spatial overlap validation
3. Preprocessing & dynamic-range normalization
4. DeepForest neural tree crown detection with sliding-window NMS
5. Coordinate transformation to projected metric CRS (m^2 / hectares)
6. Spatial union canopy dissolution & metrics calculation
7. Geospatial export generation (GeoJSON, CSV, KML, Shapefile)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.detection import (
    BaseTreeDetector,
    DeepForestDetector,
    DetectionBox,
    MockTreeDetector,
    apply_non_max_suppression,
)
from src.geometry import (
    OverlapStatus,
    OverlapValidationResult,
    check_aoi_image_overlap,
    estimate_utm_epsg,
    get_metric_projected_crs,
    pixel_box_to_geo_polygon,
    pixel_box_to_metric_polygon,
    reproject_geometry,
)
from src.io_utils import (
    GeospatialError,
    MissingGSDError,
    RasterMetadata,
    export_geospatial_results,
    load_kml_or_kmz,
    read_geotiff_metadata,
)
from src.metrics import CanopyMetrics, calculate_canopy_metrics
from src.preprocessing import clip_raster_to_polygon, normalize_to_rgb_uint8

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

try:
    from shapely.geometry import box
    HAS_SHAPELY = True
except ImportError:
    box = None
    HAS_SHAPELY = False


@dataclass
class PipelineResult:
    """Complete result container for Forest Crown AI analysis."""
    metrics: CanopyMetrics
    metadata: RasterMetadata
    detections: List[DetectionBox]
    image_rgb: np.ndarray
    aoi_validation: Optional[OverlapValidationResult] = None
    aoi_wgs84_geom: Optional[Any] = None
    crown_wgs84_geoms: List[Any] = field(default_factory=list)
    dissolved_canopy_wgs84: Optional[Any] = None
    exported_files: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def run_forest_crown_pipeline(
    image_source: Union[str, Path, bytes, io.BytesIO, np.ndarray],
    user_gsd: Optional[float] = None,
    aoi_source: Optional[Union[str, Path, bytes, io.BytesIO]] = None,
    confidence_threshold: float = 0.20,
    patch_size: int = 400,
    patch_overlap: float = 0.15,
    export_dir: Optional[Union[str, Path]] = None,
    export_base_name: str = "forest_crown_results",
    detector: Optional[BaseTreeDetector] = None,
    use_mock_detector: bool = False
) -> PipelineResult:
    """
    Executes the entire tree crown detection and canopy analysis workflow.

    Args:
        image_source: Image filepath, bytes, BytesIO, or numpy array.
        user_gsd: GSD in meters per pixel. Required for non-georeferenced images.
        aoi_source: Optional KML or KMZ boundary filepath or bytes.
        confidence_threshold: Minimum confidence score to retain crown detection.
        patch_size: Sliding window tile size in pixels.
        patch_overlap: Overlap fraction between sliding window tiles.
        export_dir: Directory to save exported GeoJSON/CSV/KML results.
        export_base_name: Prefix for output export files.
        detector: Custom detector instance (optional).
        use_mock_detector: If True, uses deterministic MockTreeDetector for testing.

    Returns:
        PipelineResult containing metrics, geometries, metadata, and export paths.

    Raises:
        MissingGSDError: If image has no georeferencing and user_gsd is None.
        GeospatialError: If AOI does not overlap image or file is invalid.
    """
    warnings_list = []

    # 1. Load image and extract spatial metadata
    if isinstance(image_source, np.ndarray):
        img_arr = image_source
        if user_gsd is None or user_gsd <= 0:
            raise MissingGSDError(
                "Numpy array input has no embedded georeferencing. "
                "Engineering Rule: Never silently invent GSD. Explicit user_gsd is required."
            )
        h, w = img_arr.shape[:2]
        c = img_arr.shape[2] if img_arr.ndim == 3 else 1
        metadata = RasterMetadata(
            width=w,
            height=h,
            count=c,
            dtype=str(img_arr.dtype),
            driver="NUMPY_ARRAY",
            crs=None,
            transform=None,
            bounds=None,
            gsd_x_m=user_gsd,
            gsd_y_m=user_gsd,
            is_geographic=False,
            is_projected=False,
            has_georeference=False,
            gsd_source="user_specified"
        )
    else:
        metadata = read_geotiff_metadata(image_source, user_gsd=user_gsd)

        # Load pixels into numpy array
        if HAS_PIL:
            if isinstance(image_source, (str, Path)):
                pil_img = Image.open(image_source)
            elif isinstance(image_source, (bytes, bytearray)):
                pil_img = Image.open(io.BytesIO(image_source))
            else:
                pil_img = Image.open(image_source)
            img_arr = np.array(pil_img)
        else:
            raise GeospatialError("PIL (Pillow) is required to decode image pixels.")

    # 2. Normalize image array for detector
    image_rgb = normalize_to_rgb_uint8(img_arr)
    gsd_m = metadata.mean_gsd_m
    assert gsd_m is not None, "GSD cannot be None after validation."

    # 3. Load and validate AOI if provided
    aoi_wgs84_geom = None
    aoi_validation = None
    aoi_img_crs = None
    offset_px = (0, 0)

    if aoi_source is not None:
        aoi_wgs84_geom, aoi_crs = load_kml_or_kmz(aoi_source)

        if metadata.has_georeference and metadata.bounds and metadata.crs:
            # Check overlap between GeoTIFF bounds and AOI
            aoi_validation = check_aoi_image_overlap(
                image_bounds=metadata.bounds,
                image_crs=metadata.crs,
                aoi_geom=aoi_wgs84_geom,
                aoi_crs=aoi_crs
            )

            if not aoi_validation.is_valid:
                raise GeospatialError(aoi_validation.warning_message or "AOI does not overlap image raster bounds.")

            if aoi_validation.warning_message:
                warnings_list.append(aoi_validation.warning_message)

            # Clip and mask raster strictly to AOI
            if metadata.transform is not None:
                # Reproject AOI to image CRS for clipping
                aoi_img_crs = reproject_geometry(aoi_wgs84_geom, aoi_crs, metadata.crs)
                image_rgb, cropped_transform, offset_px = clip_raster_to_polygon(
                    img_arr=image_rgb,
                    transform=metadata.transform,
                    polygon_geo=aoi_img_crs
                )
                if cropped_transform is not None:
                    metadata.transform = cropped_transform
                    metadata.width = image_rgb.shape[1]
                    metadata.height = image_rgb.shape[0]
                    # Update bounds to clipped image extent
                    c_x, a_x, e_y, f_y = cropped_transform[2], cropped_transform[0], cropped_transform[4], cropped_transform[5]
                    b_x0 = c_x
                    b_x1 = c_x + metadata.width * a_x
                    b_y0 = f_y + metadata.height * e_y
                    b_y1 = f_y
                    metadata.bounds = (min(b_x0, b_x1), min(b_y0, b_y1), max(b_x0, b_x1), max(b_y0, b_y1))
        else:
            warnings_list.append(
                "Image is not georeferenced (PNG/JPG). The KML boundary is preserved for visualization, "
                "but geographic pixel-level clipping cannot be performed reliably without image georeferencing. "
                "Tree detection will proceed across the full image using the user-supplied GSD."
            )

    # 4. Instantiate detector
    if detector is None:
        if use_mock_detector:
            detector = MockTreeDetector()
        else:
            try:
                detector = DeepForestDetector()
            except Exception as e:
                warnings_list.append(f"DeepForest could not be loaded ({str(e)}). Falling back to MockTreeDetector.")
                detector = MockTreeDetector()

    # 5. Run prediction on the active (clipped or full) image
    raw_detections = detector.predict(
        image_rgb=image_rgb,
        score_threshold=confidence_threshold,
        patch_size=patch_size,
        patch_overlap=patch_overlap
    )

    # 6. Apply Non-Maximum Suppression (NMS) to eliminate duplicate boundary predictions
    detections = apply_non_max_suppression(raw_detections, iou_threshold=0.30)

    # 7. Convert detections to metric and WGS84 geometries
    crown_diameters_m = []
    crown_metric_polygons = []
    crown_metric_boxes = []
    crown_wgs84_geoms = []
    crown_confidences = []
    filtered_detections = []

    # Determine metric CRS
    if metadata.has_georeference and metadata.bounds and metadata.crs:
        center_lon = (metadata.bounds[0] + metadata.bounds[2]) / 2.0
        center_lat = (metadata.bounds[1] + metadata.bounds[3]) / 2.0
        metric_crs = get_metric_projected_crs(metadata.crs, center_lon, center_lat)
        active_transform = metadata.transform
    else:
        metric_crs = "LOCAL_METRIC_CARTESIAN (meters)"
        active_transform = None

    for det in detections:
        box_coords = (det.xmin, det.ymin, det.xmax, det.ymax)

        if active_transform is not None and metadata.crs is not None:
            # Generate georeferenced box and crown ellipse using active (preserved clipped) transform
            geo_box = pixel_box_to_geo_polygon(box_coords, active_transform, as_ellipse=False)
            geo_crown = pixel_box_to_geo_polygon(box_coords, active_transform, as_ellipse=True)

            # Filter crowns strictly outside the AOI polygon if AOI exists
            if aoi_img_crs is not None and HAS_SHAPELY:
                from shapely.geometry import Point
                if not aoi_img_crs.intersects(geo_crown.centroid):
                    continue

            # Reproject to metric CRS
            metric_box = reproject_geometry(geo_box, metadata.crs, metric_crs)
            metric_crown = reproject_geometry(geo_crown, metadata.crs, metric_crs)

            if metric_box is not None:
                crown_metric_boxes.append(metric_box)
            if metric_crown is not None:
                crown_metric_polygons.append(metric_crown)

            # Reproject to WGS84 for GeoJSON/KML export and Folium maps
            if geo_crown is not None:
                wgs84_crown = reproject_geometry(geo_crown, metadata.crs, "EPSG:4326")
                crown_wgs84_geoms.append(wgs84_crown)
                det.geometry_geo = wgs84_crown
        else:
            # Non-georeferenced image with user GSD: compute directly in local meters
            metric_box = pixel_box_to_metric_polygon(box_coords, gsd_m, as_ellipse=False)
            metric_crown = pixel_box_to_metric_polygon(box_coords, gsd_m, as_ellipse=True)

            if metric_box is not None:
                crown_metric_boxes.append(metric_box)
            if metric_crown is not None:
                crown_metric_polygons.append(metric_crown)

        crown_diameters_m.append(det.approx_diameter_pixels * gsd_m)
        crown_confidences.append(det.confidence)
        det.detection_id = len(filtered_detections) + 1
        filtered_detections.append(det)

    detections = filtered_detections

    # 8. Compute AOI metric polygon for canopy cover ratio
    aoi_metric_poly = None
    if aoi_wgs84_geom is not None and metadata.has_georeference:
        aoi_metric_poly = reproject_geometry(aoi_wgs84_geom, "EPSG:4326", metric_crs)
    elif metadata.has_georeference and metadata.bounds and metadata.crs and HAS_SHAPELY:
        # If no AOI, whole image bounds is the extent
        img_box = box(*metadata.bounds)
        aoi_metric_poly = reproject_geometry(img_box, metadata.crs, metric_crs)
    elif HAS_SHAPELY:
        # Non-georeferenced full image extent in local meters
        aoi_metric_poly = box(0.0, 0.0, metadata.width * gsd_m, metadata.height * gsd_m)

    # 9. Calculate Canopy Metrics (Separates Count from Area, Dissolves Overlaps)
    metrics = calculate_canopy_metrics(
        crown_diameters_m=crown_diameters_m,
        crown_metric_polygons=crown_metric_polygons,
        crown_metric_boxes=crown_metric_boxes,
        crown_confidences=crown_confidences,
        aoi_metric_polygon=aoi_metric_poly,
        gsd_m=gsd_m,
        gsd_source=metadata.gsd_source,
        metric_crs=metric_crs,
        confidence_threshold=confidence_threshold,
        gsd_x=metadata.gsd_x_m,
        gsd_y=metadata.gsd_y_m,
        is_georeferenced=metadata.has_georeference,
        is_partial_overlap=(aoi_validation.status == OverlapStatus.PARTIAL) if aoi_validation else False
    )

    # Synchronize quality flags into warnings list
    for flag in metrics.quality_flags:
        if flag not in warnings_list:
            warnings_list.append(flag)

    # Compute dissolved canopy in WGS84 for map visualization
    dissolved_canopy_wgs84 = None
    if crown_wgs84_geoms and HAS_SHAPELY:
        from shapely.ops import unary_union
        dissolved_canopy_wgs84 = unary_union(crown_wgs84_geoms)

    # 10. Export results if export_dir provided
    exported_files = {}
    export_geoms = crown_wgs84_geoms if crown_wgs84_geoms else crown_metric_polygons
    if export_dir and export_geoms:
        export_crs = "EPSG:4326" if crown_wgs84_geoms else None
        canopy_poly = dissolved_canopy_wgs84 if crown_wgs84_geoms else None
        crown_metric_areas = [float(p.area) for p in crown_metric_polygons] if crown_metric_polygons else None
        exported_files = export_geospatial_results(
            crown_geometries=export_geoms,
            crown_scores=crown_confidences,
            canopy_polygon=canopy_poly,
            output_dir=export_dir,
            base_name=export_base_name,
            crs=export_crs,
            crown_areas_m2=crown_metric_areas
        )

    return PipelineResult(
        metrics=metrics,
        metadata=metadata,
        detections=detections,
        image_rgb=image_rgb,
        aoi_validation=aoi_validation,
        aoi_wgs84_geom=aoi_wgs84_geom,
        crown_wgs84_geoms=crown_wgs84_geoms,
        dissolved_canopy_wgs84=dissolved_canopy_wgs84,
        exported_files=exported_files,
        warnings=warnings_list
    )
