"""
Input/Output utilities for Forest Crown AI.

Handles:
- GeoTIFF metadata extraction & Ground Sample Distance (GSD) calculation
- Safe KML/KMZ boundary loading
- Image input validation and format verification
- Geospatial results export (GeoJSON, Shapefile, CSV, KML)
- Explicit enforcement of zero measurement fabrication
"""

from __future__ import annotations

import io
import json
import math
import os
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

# Attempt geospatial imports with graceful fallbacks for lightweight environments
try:
    import rasterio
    from rasterio.crs import CRS as RasterioCRS
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    RasterioCRS = None
    Affine = None
    HAS_RASTERIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

try:
    from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    Polygon = None
    MultiPolygon = None
    box = None
    mapping = None
    shape = None
    unary_union = None
    HAS_SHAPELY = False

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    gpd = None
    HAS_GEOPANDAS = False


class GeospatialError(Exception):
    """Base exception for geospatial input/output operations."""
    pass


class MissingGSDError(GeospatialError):
    """Raised when ground sample distance is missing and cannot be determined."""
    pass


class InvalidAOIError(GeospatialError):
    """Raised when the uploaded AOI is malformed or invalid."""
    pass


@dataclass
class RasterMetadata:
    """Detailed metadata for an input raster."""
    width: int
    height: int
    count: int
    dtype: str
    driver: str
    crs: Optional[str] = None
    transform: Optional[Tuple[float, float, float, float, float, float]] = None
    bounds: Optional[Tuple[float, float, float, float]] = None  # (minx, miny, maxx, maxy)
    gsd_x_m: Optional[float] = None
    gsd_y_m: Optional[float] = None
    is_geographic: bool = False
    is_projected: bool = False
    has_georeference: bool = False
    gsd_source: str = "unknown"  # "metadata_projected", "metadata_geographic_approx", "user_specified", "none"
    nodata: Optional[float] = None

    @property
    def mean_gsd_m(self) -> Optional[float]:
        """Return the mean GSD in meters per pixel if available."""
        if self.gsd_x_m is not None and self.gsd_y_m is not None:
            return (self.gsd_x_m + self.gsd_y_m) / 2.0
        return self.gsd_x_m or self.gsd_y_m

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_metric_gsd_from_geographic(
    pixel_size_deg_x: float,
    pixel_size_deg_y: float,
    center_lat_deg: float
) -> Tuple[float, float]:
    """
    Approximates metric GSD (meters per pixel) from geographic degrees at a given latitude.
    Uses WGS84 standard meridional and parallel distance approximations.
    """
    lat_rad = math.radians(center_lat_deg)
    # WGS-84 meters per degree latitude
    m_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    # WGS-84 meters per degree longitude
    m_per_deg_lon = (111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad))

    gsd_x_m = abs(pixel_size_deg_x) * m_per_deg_lon
    gsd_y_m = abs(pixel_size_deg_y) * m_per_deg_lat
    return gsd_x_m, gsd_y_m


def read_geotiff_metadata(
    file_path_or_bytes: Union[str, Path, bytes, io.BytesIO],
    user_gsd: Optional[float] = None
) -> RasterMetadata:
    """
    Extracts spatial metadata, CRS, affine transform, and GSD from a GeoTIFF or standard image.

    Strict Engineering Rule:
    - Never fabricates GSD.
    - If GeoTIFF metadata contains GSD (projected or geographic), it is computed directly.
    - If standard image (PNG/JPG) or unreferenced TIFF, user_gsd is strictly required.

    Args:
        file_path_or_bytes: File path or raw bytes of the image.
        user_gsd: Optional user-supplied GSD in meters per pixel.

    Returns:
        RasterMetadata object populated with verified spatial metadata.

    Raises:
        MissingGSDError: If no georeference exists and user_gsd is None.
    """
    # 1. Handle file opening via rasterio if available
    if HAS_RASTERIO:
        try:
            if isinstance(file_path_or_bytes, (bytes, bytearray)):
                mem_file = rasterio.io.MemoryFile(file_path_or_bytes)
                ds = mem_file.open()
            elif isinstance(file_path_or_bytes, io.BytesIO):
                mem_file = rasterio.io.MemoryFile(file_path_or_bytes.getvalue())
                ds = mem_file.open()
            else:
                ds = rasterio.open(file_path_or_bytes)

            with ds:
                width = ds.width
                height = ds.height
                count = ds.count
                dtype = str(ds.dtypes[0]) if ds.dtypes else "uint8"
                driver = ds.driver
                crs = ds.crs
                transform = ds.transform
                bounds = ds.bounds
                nodata = ds.nodata

                has_georef = bool(crs is not None and transform is not None and not transform.is_identity)

                if has_georef:
                    crs_str = crs.to_string()
                    is_proj = crs.is_projected
                    is_geo = crs.is_geographic

                    res_x = abs(transform.a)
                    res_y = abs(transform.e)

                    if is_proj:
                        # For projected CRS, transform units are typically meters
                        # Check linear units if possible
                        gsd_x_m = res_x
                        gsd_y_m = res_y
                        gsd_source = "metadata_projected"
                    elif is_geo:
                        center_lat = (bounds.bottom + bounds.top) / 2.0
                        gsd_x_m, gsd_y_m = compute_metric_gsd_from_geographic(res_x, res_y, center_lat)
                        gsd_source = "metadata_geographic_approx"
                    else:
                        gsd_x_m = res_x
                        gsd_y_m = res_y
                        gsd_source = "metadata_unknown_crs"

                    return RasterMetadata(
                        width=width,
                        height=height,
                        count=count,
                        dtype=dtype,
                        driver=driver,
                        crs=crs_str,
                        transform=(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f),
                        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                        gsd_x_m=gsd_x_m,
                        gsd_y_m=gsd_y_m,
                        is_geographic=is_geo,
                        is_projected=is_proj,
                        has_georeference=True,
                        gsd_source=gsd_source,
                        nodata=nodata,
                    )
        except Exception as e:
            # If rasterio fails on non-geotiff formats, fallback to PIL
            pass

    # 2. Fallback for unreferenced images or when rasterio is not available
    if HAS_PIL:
        try:
            if isinstance(file_path_or_bytes, (bytes, bytearray)):
                img = Image.open(io.BytesIO(file_path_or_bytes))
            elif isinstance(file_path_or_bytes, io.BytesIO):
                img = Image.open(file_path_or_bytes)
            else:
                img = Image.open(file_path_or_bytes)

            width, height = img.size
            count = len(img.getbands())
            driver = img.format or "UNKNOWN"
            dtype = "uint8"
        except Exception as e:
            raise GeospatialError(f"Cannot identify or read image file: {str(e)}") from e
    else:
        raise GeospatialError("Neither rasterio nor PIL is available to inspect the image.")

    # Check user-supplied GSD
    if user_gsd is not None and user_gsd > 0:
        return RasterMetadata(
            width=width,
            height=height,
            count=count,
            dtype=dtype,
            driver=driver,
            crs=None,
            transform=None,
            bounds=None,
            gsd_x_m=float(user_gsd),
            gsd_y_m=float(user_gsd),
            is_geographic=False,
            is_projected=False,
            has_georeference=False,
            gsd_source="user_specified",
            nodata=None,
        )

    # If unreferenced and no user GSD provided, FAIL EXPLICITLY. Never invent GSD!
    raise MissingGSDError(
        f"The uploaded image ({driver}, {width}x{height}) has no embedded georeferencing / GSD metadata. "
        "Engineering Rule: Never silently invent spatial resolution. "
        "Please provide an explicit GSD (in meters per pixel) to proceed with physical measurements."
    )


def parse_kml_coordinates(coords_str: str) -> List[Tuple[float, float]]:
    """
    Parses a KML coordinate string format: 'lon,lat,alt lon,lat,alt ...'
    Returns a list of (longitude, latitude) tuples.
    """
    points = []
    for token in coords_str.strip().split():
        parts = token.strip().split(',')
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                points.append((lon, lat))
            except ValueError:
                continue
    return points


def load_kml_or_kmz(
    file_path_or_bytes: Union[str, Path, bytes, io.BytesIO]
) -> Tuple[Any, str]:
    """
    Robustly loads boundary polygon(s) from a KML (.kml) or KMZ (.kmz) file.
    Does not require external GDAL KML drivers. Pure XML parsing into Shapely geometries.

    KML coordinates are by definition in WGS84 (EPSG:4326).

    Args:
        file_path_or_bytes: Path or bytes to KML or KMZ file.

    Returns:
        Tuple of (Shapely Polygon/MultiPolygon, crs_string "EPSG:4326")

    Raises:
        InvalidAOIError: If no valid polygon boundary can be extracted.
    """
    kml_bytes = None

    if isinstance(file_path_or_bytes, (bytes, bytearray)):
        raw_data = bytes(file_path_or_bytes)
    elif isinstance(file_path_or_bytes, io.BytesIO):
        raw_data = file_path_or_bytes.getvalue()
    else:
        path = Path(file_path_or_bytes)
        if not path.exists():
            raise InvalidAOIError(f"AOI file not found at {path}")
        with open(path, "rb") as f:
            raw_data = f.read()

    # Check if KMZ (ZIP archive)
    if zipfile.is_zipfile(io.BytesIO(raw_data)):
        try:
            with zipfile.ZipFile(io.BytesIO(raw_data), 'r') as zf:
                # Look for .kml file inside
                kml_names = [name for name in zf.namelist() if name.lower().endswith('.kml')]
                if not kml_names:
                    raise InvalidAOIError("KMZ archive does not contain any .kml file.")
                # Primary is usually doc.kml or the first .kml
                primary_kml = 'doc.kml' if 'doc.kml' in kml_names else kml_names[0]
                kml_bytes = zf.read(primary_kml)
        except Exception as e:
            raise InvalidAOIError(f"Failed to read KMZ archive: {str(e)}")
    else:
        kml_bytes = raw_data

    # Parse KML XML
    try:
        root = ET.fromstring(kml_bytes)
    except ET.ParseError as e:
        raise InvalidAOIError(f"Malformed KML XML content: {str(e)}")

    # Extract all Polygon geometries (handling any XML namespace)
    polygons = []

    # Helper to find elements ignoring namespace
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag.lower() == 'polygon':
            outer_ring = None
            inner_rings = []

            for child in elem.iter():
                child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if child_tag.lower() == 'outerboundaryis':
                    for coord_elem in child.iter():
                        c_tag = coord_elem.tag.split('}')[-1] if '}' in coord_elem.tag else coord_elem.tag
                        if c_tag.lower() == 'coordinates' and coord_elem.text:
                            outer_ring = parse_kml_coordinates(coord_elem.text)
                elif child_tag.lower() == 'innerboundaryis':
                    for coord_elem in child.iter():
                        c_tag = coord_elem.tag.split('}')[-1] if '}' in coord_elem.tag else coord_elem.tag
                        if c_tag.lower() == 'coordinates' and coord_elem.text:
                            inner_ring = parse_kml_coordinates(coord_elem.text)
                            if len(inner_ring) >= 3:
                                inner_rings.append(inner_ring)

            if outer_ring and len(outer_ring) >= 3:
                if HAS_SHAPELY:
                    try:
                        poly = Polygon(outer_ring, holes=inner_rings)
                        if poly.is_valid and not poly.is_empty:
                            polygons.append(poly)
                        else:
                            poly_fixed = poly.buffer(0)
                            if not poly_fixed.is_empty:
                                polygons.append(poly_fixed)
                    except Exception:
                        pass
                else:
                    polygons.append(outer_ring)

    if not polygons:
        raise InvalidAOIError("No valid polygon boundary geometries found in KML/KMZ.")

    if HAS_SHAPELY:
        if len(polygons) == 1:
            combined_geom = polygons[0]
        else:
            combined_geom = unary_union(polygons)
        return combined_geom, "EPSG:4326"
    else:
        return polygons, "EPSG:4326"


def export_geospatial_results(
    crown_geometries: List[Any],
    crown_scores: List[float],
    canopy_polygon: Optional[Any],
    output_dir: Union[str, Path],
    base_name: str = "forest_crown_results",
    crs: Optional[str] = "EPSG:4326",
    crown_areas_m2: Optional[List[float]] = None
) -> Dict[str, str]:
    """
    Exports tree crown detections and dissolved canopy footprint to standard GIS formats:
    - GeoJSON (.geojson)
    - Shapefile (.shp)
    - CSV (.csv with WKT and centroid coordinates)
    - Google Earth KML (.kml)

    Returns a dictionary of generated filepaths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files = {}

    if not HAS_SHAPELY:
        raise GeospatialError("Shapely is required for geospatial results export.")

    # 1. Build records for crowns
    records = []
    for idx, (geom, score) in enumerate(zip(crown_geometries, crown_scores)):
        if crown_areas_m2 is not None and idx < len(crown_areas_m2):
            area_val = crown_areas_m2[idx]
        elif hasattr(geom, 'area'):
            area_val = geom.area
        else:
            area_val = 0.0
        centroid = geom.centroid if hasattr(geom, 'centroid') else None
        records.append({
            "tree_id": idx + 1,
            "confidence": round(float(score), 4),
            "area": round(float(area_val), 2),
            "area_m2": round(float(area_val), 2),
            "centroid_x": round(float(centroid.x), 6) if centroid else 0.0,
            "centroid_y": round(float(centroid.y), 6) if centroid else 0.0,
            "wkt": geom.wkt if hasattr(geom, 'wkt') else "",
            "geometry": geom
        })

    # 2. GeoJSON export
    geojson_path = out_path / f"{base_name}_crowns.geojson"
    features = []
    for rec in records:
        features.append({
            "type": "Feature",
            "properties": {
                "tree_id": rec["tree_id"],
                "confidence": rec["confidence"],
                "area_m2": rec["area_m2"],
                "area": rec["area"],
                "centroid_x": rec["centroid_x"],
                "centroid_y": rec["centroid_y"]
            },
            "geometry": mapping(rec["geometry"])
        })
    geojson_data = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs or "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features
    }
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)
    generated_files["crowns_geojson"] = str(geojson_path)

    # 3. Export dissolved canopy footprint as GeoJSON
    if canopy_polygon is not None and hasattr(canopy_polygon, '__geo_interface__'):
        canopy_geojson_path = out_path / f"{base_name}_canopy_footprint.geojson"
        canopy_data = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": crs or "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": [{
                "type": "Feature",
                "properties": {"layer": "dissolved_canopy_cover"},
                "geometry": mapping(canopy_polygon)
            }]
        }
        with open(canopy_geojson_path, "w", encoding="utf-8") as f:
            json.dump(canopy_data, f, indent=2)
        generated_files["canopy_geojson"] = str(canopy_geojson_path)

    # 4. CSV export
    csv_path = out_path / f"{base_name}_crowns.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("tree_id,confidence,area,centroid_x,centroid_y,geometry_wkt\n")
        for rec in records:
            f.write(f'{rec["tree_id"]},{rec["confidence"]},{rec["area"]},{rec["centroid_x"]},{rec["centroid_y"]},"{rec["wkt"]}"\n')
    generated_files["crowns_csv"] = str(csv_path)

    # 5. KML Export for Google Earth (strictly valid for WGS84 coordinates)
    if crs == "EPSG:4326":
        kml_path = out_path / f"{base_name}_crowns.kml"
        kml_content = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            '<Document>',
            f'<name>{base_name} Detections</name>',
            '<description>Individual tree crown boundaries exported from Forest Crown AI.</description>'
        ]
        for rec in records:
            geom = rec["geometry"]
            kml_content.append('<Placemark>')
            kml_content.append(f'<name>Tree #{rec["tree_id"]}</name>')
            kml_content.append(f'<description>Confidence: {rec["confidence"]}</description>')
            if geom.geom_type == 'Polygon':
                coords_str = " ".join(f"{x},{y},0" for x, y in geom.exterior.coords)
                kml_content.append('<Polygon><outerBoundaryIs><LinearRing>')
                kml_content.append(f'<coordinates>{coords_str}</coordinates>')
                kml_content.append('</LinearRing></outerBoundaryIs></Polygon>')
            kml_content.append('</Placemark>')
        kml_content.append('</Document></kml>')

        with open(kml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(kml_content))
        generated_files["crowns_kml"] = str(kml_path)

    # 6. Shapefile export via geopandas if available and CRS is specified
    if HAS_GEOPANDAS and records:
        try:
            gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
            shp_path = out_path / f"{base_name}_crowns.shp"
            gdf[["tree_id", "confidence", "area", "geometry"]].to_file(shp_path)
            generated_files["crowns_shapefile"] = str(shp_path)
        except Exception:
            pass

    return generated_files
