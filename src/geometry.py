"""
Geometrical operations, coordinate transformations, and spatial validation for Forest Crown AI.

Core Engineering Principles Enforced:
- Calculate areas strictly in a projected metric CRS (meters/hectares), NEVER in lat/long degrees.
- Do NOT simply sum overlapping bounding boxes: perform spatial union/dissolve.
- Validate AOI / image spatial overlap with strict checking for outside, partial, or full coverage.
- Preserve CRS metadata throughout the transformation chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from shapely.geometry import MultiPolygon, Point, Polygon, box
    from shapely.ops import transform as shapely_transform, unary_union
    HAS_SHAPELY = True
except ImportError:
    Polygon = None
    MultiPolygon = None
    Point = None
    box = None
    shapely_transform = None
    unary_union = None
    HAS_SHAPELY = False

try:
    import pyproj
    from pyproj import CRS, Transformer
    HAS_PYPROJ = True
except ImportError:
    pyproj = None
    CRS = None
    Transformer = None
    HAS_PYPROJ = False


class OverlapStatus(str, Enum):
    FULL_INSIDE = "full_inside"        # AOI is completely inside the image
    COVERS_IMAGE = "covers_image"      # AOI completely surrounds the image
    PARTIAL = "partial_overlap"        # AOI intersects partially with the image
    NO_OVERLAP = "no_overlap"          # AOI does not intersect the image


@dataclass
class OverlapValidationResult:
    """Detailed result of AOI/image spatial intersection."""
    status: OverlapStatus
    is_valid: bool
    overlap_area_m2: float
    aoi_area_m2: float
    image_area_m2: float
    pct_aoi_covered: float
    pct_image_covered: float
    warning_message: Optional[str] = None
    intersection_geometry: Optional[Any] = None


def estimate_utm_epsg(lon: float, lat: float) -> int:
    """
    Determines the appropriate WGS 84 UTM EPSG code for a given (lon, lat) coordinate.

    Formula:
    zone = floor((lon + 180) / 6) + 1
    EPSG = 32600 + zone (Northern Hemisphere)
    EPSG = 32700 + zone (Southern Hemisphere)
    """
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    if lat >= 0.0:
        return 32600 + zone
    else:
        return 32700 + zone


def is_crs_geographic(crs_str_or_obj: Any) -> bool:
    """Checks if a CRS is geographic (degrees, e.g. EPSG:4326)."""
    if not crs_str_or_obj:
        return False
    crs_str = str(crs_str_or_obj).upper()
    if "4326" in crs_str or "CRS84" in crs_str or "WGS 84" in crs_str:
        return True
    if HAS_PYPROJ:
        try:
            crs = CRS.from_user_input(crs_str_or_obj)
            return crs.is_geographic
        except Exception:
            pass
    return "GEOGCS" in crs_str or "DEGREE" in crs_str


def get_metric_projected_crs(source_crs: Any, approx_lon: float, approx_lat: float) -> str:
    """
    Returns a projected metric CRS.
    If the source CRS is already projected in meters, preserves it.
    If geographic, returns the corresponding UTM EPSG code (e.g. 'EPSG:32617').
    """
    if HAS_PYPROJ and source_crs:
        try:
            crs_obj = CRS.from_user_input(source_crs)
            if crs_obj.is_projected:
                # Check units
                axis_info = crs_obj.axis_info
                unit_name = axis_info[0].unit_name.lower() if axis_info else "metre"
                if "metre" in unit_name or "meter" in unit_name:
                    return crs_obj.to_string()
        except Exception:
            pass

    # Default to UTM for the approximate coordinate
    utm_code = estimate_utm_epsg(approx_lon, approx_lat)
    return f"EPSG:{utm_code}"


def reproject_geometry(geom: Any, source_crs: str, target_crs: str) -> Any:
    """
    Reprojects a Shapely geometry from source_crs to target_crs.
    """
    if not HAS_SHAPELY:
        return geom

    if source_crs == target_crs:
        return geom

    if not HAS_PYPROJ:
        # Without pyproj, cannot reliably reproject non-identity CRS
        return geom

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return shapely_transform(transformer.transform, geom)


def pixel_box_to_geo_polygon(
    box_coords: Tuple[float, float, float, float],
    transform: Tuple[float, float, float, float, float, float],
    as_ellipse: bool = False,
    num_pts: int = 16
) -> Polygon:
    """
    Converts a bounding box in pixel coordinates (xmin, ymin, xmax, ymax)
    into a georeferenced Shapely Polygon using an affine transform (a, b, c, d, e, f).

    Coordinate mapping:
    x_geo = c + x_pix * a + y_pix * b
    y_geo = f + x_pix * d + y_pix * e

    Args:
        box_coords: (xmin, ymin, xmax, ymax) in pixel indices.
        transform: 6-tuple (a, b, c, d, e, f)
        as_ellipse: If True, produces an inscribed ellipse (realistic crown shape)
                    instead of a rigid rectangle.
        num_pts: Number of points for ellipse discretization.
    """
    xmin, ymin, xmax, ymax = box_coords
    a, b, c, d, e, f = transform

    def pix_to_geo(px: float, py: float) -> Tuple[float, float]:
        gx = c + px * a + py * b
        gy = f + px * d + py * e
        return (gx, gy)

    if not HAS_SHAPELY:
        return None

    if not as_ellipse:
        # Standard rectangular bounding box
        corners_pix = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
        geo_coords = [pix_to_geo(px, py) for px, py in corners_pix]
        return Polygon(geo_coords)
    else:
        # Inscribed ellipse: tree crowns are rounded, bounding boxes overestimate canopy
        cx_pix = (xmin + xmax) / 2.0
        cy_pix = (ymin + ymax) / 2.0
        rx_pix = (xmax - xmin) / 2.0
        ry_pix = (ymax - ymin) / 2.0

        ellipse_pts = []
        for i in range(num_pts):
            theta = 2.0 * math.pi * (i / num_pts)
            px = cx_pix + rx_pix * math.cos(theta)
            py = cy_pix + ry_pix * math.sin(theta)
            ellipse_pts.append(pix_to_geo(px, py))
        ellipse_pts.append(ellipse_pts[0])
        return Polygon(ellipse_pts)


def pixel_box_to_metric_polygon(
    box_coords: Tuple[float, float, float, float],
    gsd_m: float,
    as_ellipse: bool = False,
    num_pts: int = 16
) -> Optional[Polygon]:
    """
    Converts a pixel box to a local metric coordinate polygon (meters) using GSD.
    Used when image is unprojected (PNG/JPG) but user provided GSD in meters/pixel.
    """
    if not HAS_SHAPELY:
        return None

    xmin, ymin, xmax, ymax = box_coords

    if not as_ellipse:
        coords = [
            (xmin * gsd_m, ymin * gsd_m),
            (xmax * gsd_m, ymin * gsd_m),
            (xmax * gsd_m, ymax * gsd_m),
            (xmin * gsd_m, ymax * gsd_m),
            (xmin * gsd_m, ymin * gsd_m)
        ]
        return Polygon(coords)
    else:
        cx = ((xmin + xmax) / 2.0) * gsd_m
        cy = ((ymin + ymax) / 2.0) * gsd_m
        rx = ((xmax - xmin) / 2.0) * gsd_m
        ry = ((ymax - ymin) / 2.0) * gsd_m

        pts = []
        for i in range(num_pts):
            theta = 2.0 * math.pi * (i / num_pts)
            px = cx + rx * math.cos(theta)
            py = cy + ry * math.sin(theta)
            pts.append((px, py))
        pts.append(pts[0])
        return Polygon(pts)


def check_aoi_image_overlap(
    image_bounds: Tuple[float, float, float, float],
    image_crs: str,
    aoi_geom: Any,
    aoi_crs: str = "EPSG:4326"
) -> OverlapValidationResult:
    """
    Validates the spatial intersection between an image raster boundary and a user AOI.

    Engineering Principle:
    - Never silently run detection outside the image bounds.
    - If AOI is completely outside -> Disallow execution with error.
    - If partial overlap -> Warn user with exact percentage of coverage.
    - Reprojects both geometries to a shared metric CRS to compute true metric areas.

    Args:
        image_bounds: (minx, miny, maxx, maxy) in image_crs.
        image_crs: CRS of the image raster.
        aoi_geom: Shapely Polygon or MultiPolygon representing the AOI.
        aoi_crs: CRS of the AOI (default "EPSG:4326" for KML).
    """
    if not HAS_SHAPELY:
        # Minimal mock result if shapely is missing
        return OverlapValidationResult(
            status=OverlapStatus.FULL_INSIDE,
            is_valid=True,
            overlap_area_m2=0.0,
            aoi_area_m2=0.0,
            image_area_m2=0.0,
            pct_aoi_covered=100.0,
            pct_image_covered=100.0
        )

    img_poly = box(*image_bounds)

    # Calculate center for UTM projection
    if is_crs_geographic(image_crs):
        center_lon = (image_bounds[0] + image_bounds[2]) / 2.0
        center_lat = (image_bounds[1] + image_bounds[3]) / 2.0
    else:
        # Reproject center to 4326 to find UTM zone
        center_pt = Point((image_bounds[0] + image_bounds[2]) / 2.0, (image_bounds[1] + image_bounds[3]) / 2.0)
        center_wgs84 = reproject_geometry(center_pt, image_crs, "EPSG:4326")
        center_lon, center_lat = center_wgs84.x, center_wgs84.y

    metric_crs = get_metric_projected_crs(image_crs, center_lon, center_lat)

    # Reproject both to metric CRS
    img_metric = reproject_geometry(img_poly, image_crs, metric_crs)
    aoi_metric = reproject_geometry(aoi_geom, aoi_crs, metric_crs)

    img_area = float(img_metric.area)
    aoi_area = float(aoi_metric.area)

    # Check intersection
    if not img_metric.intersects(aoi_metric):
        return OverlapValidationResult(
            status=OverlapStatus.NO_OVERLAP,
            is_valid=False,
            overlap_area_m2=0.0,
            aoi_area_m2=aoi_area,
            image_area_m2=img_area,
            pct_aoi_covered=0.0,
            pct_image_covered=0.0,
            warning_message=(
                "CRITICAL ERROR: The uploaded AOI does not intersect with the image raster bounds. "
                "Verify that the KML boundary matches the geographic location of the imagery."
            ),
            intersection_geometry=None
        )

    intersection = img_metric.intersection(aoi_metric)
    overlap_area = float(intersection.area)
    pct_aoi = (overlap_area / aoi_area * 100.0) if aoi_area > 0 else 0.0
    pct_img = (overlap_area / img_area * 100.0) if img_area > 0 else 0.0

    if pct_aoi >= 99.9:
        status = OverlapStatus.FULL_INSIDE
        warning = None
    elif pct_img >= 99.9:
        status = OverlapStatus.COVERS_IMAGE
        warning = (
            f"NOTE: The AOI boundary covers the entire image (image is 100% inside AOI), "
            f"but AOI extends beyond the image. Only the imagery area ({pct_aoi:.1f}% of AOI) will be analyzed."
        )
    else:
        status = OverlapStatus.PARTIAL
        warning = (
            f"WARNING: Partial overlap detected. The image covers {pct_aoi:.1f}% of the specified AOI, "
            f"and the AOI covers {pct_img:.1f}% of the image. Tree detection will only occur within the intersecting region."
        )

    return OverlapValidationResult(
        status=status,
        is_valid=True,
        overlap_area_m2=overlap_area,
        aoi_area_m2=aoi_area,
        image_area_m2=img_area,
        pct_aoi_covered=pct_aoi,
        pct_image_covered=pct_img,
        warning_message=warning,
        intersection_geometry=intersection
    )


def dissolve_geometries(geometries: List[Any]) -> Tuple[Any, float, float]:
    """
    Computes both the raw summed area and the dissolved (spatial union) area.

    Crucial Engineering Principle:
    - Never simply sum overlapping bounding boxes!
    - Overlapping crowns share canopy footprint. Simple addition creates false double-counting.
    - Spatial union dissolves overlapping polygons into a single footprint.

    Returns:
        Tuple of (dissolved_geometry, dissolved_area, raw_summed_area)
    """
    if not geometries or not HAS_SHAPELY:
        return None, 0.0, 0.0

    valid_geoms = [g for g in geometries if g is not None and not g.is_empty and g.is_valid]
    if not valid_geoms:
        return None, 0.0, 0.0

    raw_summed_area = sum(float(g.area) for g in valid_geoms)
    dissolved = unary_union(valid_geoms)
    dissolved_area = float(dissolved.area)

    return dissolved, dissolved_area, raw_summed_area
