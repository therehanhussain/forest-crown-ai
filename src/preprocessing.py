"""
Preprocessing, image normalization, AOI clipping, and windowed tiling for Forest Crown AI.

Prepares aerial and drone imagery for DeepForest tree crown detection:
- Validates formats, channels, and spatial attributes.
- Normalizes multispectral, 16-bit, or RGBA rasters into 3-channel RGB uint8.
- Clips raster to AOI boundary polygons.
- Handles sliding-window tiling with configurable seam overlap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Tuple, Union

import numpy as np

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    Image = None
    ImageDraw = None
    HAS_PIL = False

try:
    from shapely.geometry import MultiPolygon, Polygon, box
    HAS_SHAPELY = True
except ImportError:
    Polygon = None
    MultiPolygon = None
    box = None
    HAS_SHAPELY = False

try:
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    geometry_mask = None
    Affine = None
    HAS_RASTERIO = False


@dataclass
class PreprocessingValidationResult:
    """Validation report on an input image."""
    is_valid: bool
    width: int
    height: int
    channels: int
    dtype: str
    warnings: List[str]
    errors: List[str]


def validate_image_array(img_arr: np.ndarray) -> PreprocessingValidationResult:
    """
    Validates dimensions, channels, and data types of an image array.
    """
    errors = []
    warnings = []

    if img_arr.ndim == 2:
        height, width = img_arr.shape
        channels = 1
        warnings.append("Single-channel grayscale image detected; will replicate across 3 RGB channels.")
    elif img_arr.ndim == 3:
        # Check channel position (H, W, C) vs (C, H, W)
        if img_arr.shape[0] in (1, 3, 4) and img_arr.shape[2] not in (1, 3, 4):
            # Probably (C, H, W) from rasterio
            channels, height, width = img_arr.shape
            warnings.append("Rasterio channel-first format detected; will transpose to (H, W, C).")
        else:
            height, width, channels = img_arr.shape
    else:
        return PreprocessingValidationResult(
            is_valid=False,
            width=0,
            height=0,
            channels=0,
            dtype=str(img_arr.dtype),
            warnings=[],
            errors=[f"Unsupported image array dimensions: {img_arr.ndim} (expected 2 or 3)."]
        )

    if width < 50 or height < 50:
        errors.append(f"Image too small ({width}x{height}). Minimum required is 50x50 pixels.")

    if channels > 4:
        warnings.append(f"Multispectral image with {channels} bands detected. First 3 bands will be used as RGB.")

    dtype_str = str(img_arr.dtype)
    if dtype_str != "uint8":
        warnings.append(f"Image dtype is {dtype_str}. Will apply 2-98% percentile visual stretch to uint8 [0, 255].")

    is_valid = len(errors) == 0
    return PreprocessingValidationResult(
        is_valid=is_valid,
        width=width,
        height=height,
        channels=channels,
        dtype=dtype_str,
        warnings=warnings,
        errors=errors
    )


def normalize_to_rgb_uint8(img_arr: np.ndarray) -> np.ndarray:
    """
    Normalizes any image array into a 3-channel (H, W, 3) uint8 [0, 255] RGB array.
    Applies 2%-98% percentile stretching for high-dynamic-range (16-bit / float) imagery.
    Removes alpha channel or duplicates single channel if necessary.
    """
    # 1. Handle (C, H, W) -> (H, W, C)
    if img_arr.ndim == 3 and img_arr.shape[0] in (1, 3, 4, 5, 6, 7, 8) and img_arr.shape[2] > 10:
        img_arr = np.transpose(img_arr, (1, 2, 0))

    # 2. Extract 3 channels
    if img_arr.ndim == 2:
        # Grayscale -> 3 channel
        img_arr = np.stack([img_arr, img_arr, img_arr], axis=-1)
    elif img_arr.ndim == 3:
        if img_arr.shape[-1] == 1:
            img_arr = np.repeat(img_arr, 3, axis=-1)
        elif img_arr.shape[-1] == 4:
            # RGBA: extract RGB
            img_arr = img_arr[:, :, :3]
        elif img_arr.shape[-1] > 3:
            # Multispectral: take first 3 bands
            img_arr = img_arr[:, :, :3]

    # 3. Dynamic range normalization
    if img_arr.dtype == np.uint8:
        return img_arr

    # Percentile contrast stretch to handle 16-bit, float32, etc.
    out = np.zeros(img_arr.shape, dtype=np.uint8)
    for c in range(3):
        band = img_arr[:, :, c]
        valid_mask = np.isfinite(band) & (band > 0)
        if np.any(valid_mask):
            p2, p98 = np.percentile(band[valid_mask], (2, 98))
            if p98 > p2:
                scaled = (band - p2) / (p98 - p2) * 255.0
                out[:, :, c] = np.clip(scaled, 0, 255).astype(np.uint8)
            else:
                out[:, :, c] = np.clip(band, 0, 255).astype(np.uint8)
        else:
            out[:, :, c] = 0

    return out


@dataclass
class WindowTile:
    """Represents an image sub-tile for sliding-window inference."""
    col_off: int
    row_off: int
    width: int
    height: int
    tile_array: np.ndarray


def generate_sliding_window_tiles(
    img_arr: np.ndarray,
    tile_size: int = 400,
    overlap_ratio: float = 0.15
) -> Generator[WindowTile, None, None]:
    """
    Generates overlapping tiles for large orthomosaic imagery.

    Args:
        img_arr: (H, W, 3) uint8 image array.
        tile_size: Window dimension in pixels (default 400 for DeepForest).
        overlap_ratio: Overlap fraction between adjacent tiles (e.g. 0.15 = 15%).

    Yields:
        WindowTile instances with offsets and sliced sub-arrays.
    """
    h, w = img_arr.shape[:2]
    step = int(tile_size * (1.0 - overlap_ratio))
    step = max(1, step)

    for y in range(0, h, step):
        for x in range(0, w, step):
            actual_w = min(tile_size, w - x)
            actual_h = min(tile_size, h - y)

            tile_slice = img_arr[y:y + actual_h, x:x + actual_w]

            # If tile is smaller than tile_size at boundaries, pad to square if needed
            yield WindowTile(
                col_off=x,
                row_off=y,
                width=actual_w,
                height=actual_h,
                tile_array=tile_slice
            )


def clip_raster_to_polygon(
    img_arr: np.ndarray,
    transform: Tuple[float, float, float, float, float, float],
    polygon_geo: Any
) -> Tuple[np.ndarray, Tuple[float, float, float, float, float, float], Tuple[int, int]]:
    """
    Clips an image array to a georeferenced polygon.
    Sets all pixels outside the polygon to zero (black).

    Returns:
        Tuple of (clipped_array, new_transform, (col_offset, row_offset))
    """
    if not HAS_SHAPELY or polygon_geo is None:
        return img_arr, transform, (0, 0)

    a, b, c, d, e, f = transform
    # Inverse affine transform to map geo -> pixel
    # In pure form:
    # px = (e*(gx - c) - b*(gy - f)) / (a*e - b*d)
    # py = (-d*(gx - c) + a*(gy - f)) / (a*e - b*d)
    det = a * e - b * d
    if abs(det) < 1e-12:
        return img_arr, transform, (0, 0)

    minx, miny, maxx, maxy = polygon_geo.bounds

    def geo_to_pix(gx: float, gy: float) -> Tuple[float, float]:
        px = (e * (gx - c) - b * (gy - f)) / det
        py = (-d * (gx - c) + a * (gy - f)) / det
        return px, py

    p1 = geo_to_pix(minx, maxy)
    p2 = geo_to_pix(maxx, miny)

    px_min = max(0, int(math.floor(min(p1[0], p2[0]))))
    py_min = max(0, int(math.floor(min(p1[1], p2[1]))))
    px_max = min(img_arr.shape[1], int(math.ceil(max(p1[0], p2[0]))))
    py_max = min(img_arr.shape[0], int(math.ceil(max(p1[1], p2[1]))))

    if px_max <= px_min or py_max <= py_min:
        return img_arr, transform, (0, 0)

    cropped = img_arr[py_min:py_max, px_min:px_max].copy()

    # Update affine transform for cropped origin
    new_c = c + px_min * a + py_min * b
    new_f = f + px_min * d + py_min * e
    new_transform = (a, b, new_c, d, e, new_f)

    # Strictly mask out pixels outside the polygon (handling holes and MultiPolygons)
    try:
        if HAS_RASTERIO and geometry_mask is not None and Affine is not None:
            affine_cropped = Affine(a, b, new_c, d, e, new_f)
            mask = geometry_mask(
                [polygon_geo],
                transform=affine_cropped,
                invert=True,
                out_shape=(cropped.shape[0], cropped.shape[1])
            )
            cropped[~mask] = 0
        elif HAS_PIL and ImageDraw is not None:
            mask_img = Image.new("1", (cropped.shape[1], cropped.shape[0]), 0)
            draw = ImageDraw.Draw(mask_img)

            def geo_to_cropped_pix(gx: float, gy: float) -> Tuple[float, float]:
                px = (e * (gx - new_c) - b * (gy - new_f)) / det
                py = (-d * (gx - new_c) + a * (gy - new_f)) / det
                return (px, py)

            geoms = polygon_geo.geoms if hasattr(polygon_geo, "geoms") else [polygon_geo]
            for poly in geoms:
                if hasattr(poly, "exterior") and poly.exterior:
                    coords = [geo_to_cropped_pix(x, y) for x, y in poly.exterior.coords]
                    draw.polygon(coords, outline=1, fill=1)
                    if hasattr(poly, "interiors"):
                        for interior in poly.interiors:
                            int_coords = [geo_to_cropped_pix(x, y) for x, y in interior.coords]
                            draw.polygon(int_coords, outline=0, fill=0)

            mask_arr = np.array(mask_img, dtype=bool)
            cropped[~mask_arr] = 0
    except Exception:
        # Fallback: cropped bounding box preserved if masking encounter an issue
        pass

    return cropped, new_transform, (px_min, py_min)
