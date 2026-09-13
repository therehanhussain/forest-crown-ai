"""
Smoke Test for DeepForest Integration.

Verifies end-to-end execution of the real pretrained DeepForest neural network model
WITHOUT any mocks, using official weights and a real RGB aerial forest image (OSBS_029.png).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import run_forest_crown_pipeline
from src.detection import (
    DeepForestDetector,
    DetectionBox,
    load_deepforest_model,
)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import deepforest
    from deepforest import main as df_main
    HAS_DEEPFOREST = True
except ImportError:
    HAS_DEEPFOREST = False


def run_smoke_test():
    print("=" * 70)
    print("FOREST CROWN AI - REAL DEEPFOREST SMOKE TEST (ZERO MOCKING)")
    print("=" * 70)

    # 1. Environment & Package Verification
    print(f"[*] Python version: {sys.version.split()[0]}")
    assert HAS_DEEPFOREST, "DeepForest is not installed in the environment!"
    print(f"[*] DeepForest version: {deepforest.__version__}")

    # 2. Locate Real Sample Forest Imagery
    sample_image_path = deepforest.get_data("OSBS_029.png")
    assert os.path.exists(sample_image_path), f"Sample image not found: {sample_image_path}"
    print(f"[*] Real aerial test image: {sample_image_path}")

    pil_img = Image.open(sample_image_path)
    img_rgb = np.array(pil_img.convert("RGB"))
    h, w, c = img_rgb.shape
    print(f"[*] Image dimensions: {w}x{h} px, {c} channels, dtype={img_rgb.dtype}")

    # 3. Measure Model Initialization Time
    print("\n[*] Initializing DeepForest model and loading release weights...")
    t_init_start = time.perf_counter()
    model = load_deepforest_model()
    t_init_end = time.perf_counter()
    init_duration = t_init_end - t_init_start
    print(f"[OK] Model loaded in {init_duration:.3f} seconds.")

    # 4. Initialize DeepForestDetector with the loaded model instance
    detector = DeepForestDetector(model_instance=model)

    # 5. Measure Model Inference Time
    print("\n[*] Executing real neural inference (score_threshold=0.20)...")
    t_inf_start = time.perf_counter()
    detections = detector.predict(
        image_rgb=img_rgb,
        score_threshold=0.20,
        patch_size=400,
        patch_overlap=0.15
    )
    t_inf_end = time.perf_counter()
    inf_duration = t_inf_end - t_inf_start
    print(f"[OK] Inference executed in {inf_duration:.3f} seconds.")

    # 6. Verify Detections and Bounding Box Coordinates
    print(f"\n[*] Total tree crowns detected: {len(detections)}")
    assert len(detections) > 0, "Expected at least one tree crown detection on OSBS_029.png!"

    for i, det in enumerate(detections):
        # Validate coordinate bounds
        assert 0.0 <= det.xmin < det.xmax <= w + 1.0, f"Invalid x-coords in box {i}: xmin={det.xmin}, xmax={det.xmax}"
        assert 0.0 <= det.ymin < det.ymax <= h + 1.0, f"Invalid y-coords in box {i}: ymin={det.ymin}, ymax={det.ymax}"
        assert 0.0 <= det.confidence <= 1.0, f"Invalid confidence in box {i}: {det.confidence}"
        assert det.label == "Tree", f"Unexpected label in box {i}: {det.label}"

    print(f"[OK] All {len(detections)} bounding boxes verified (xmin < xmax, ymin < ymax, confidence in [0, 1]).")
    sample_box = detections[0]
    print(f"    Sample detection: xmin={sample_box.xmin:.1f}, ymin={sample_box.ymin:.1f}, "
          f"xmax={sample_box.xmax:.1f}, ymax={sample_box.ymax:.1f}, conf={sample_box.confidence:.3f}")

    # 7. End-to-End Pipeline Execution with Metric Geometry & Dissolution
    print("\n[*] Executing End-to-End Pipeline with DeepForestDetector (GSD=0.10 m/px)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        t_pipe_start = time.perf_counter()
        pipeline_result = run_forest_crown_pipeline(
            image_source=sample_image_path,
            user_gsd=0.10,
            confidence_threshold=0.20,
            patch_size=400,
            patch_overlap=0.15,
            export_dir=tmpdir,
            export_base_name="smoke_test_run",
            detector=detector,
            use_mock_detector=False
        )
        t_pipe_end = time.perf_counter()

        m = pipeline_result.metrics
        print(f"[OK] Pipeline completed in {t_pipe_end - t_pipe_start:.3f} seconds.")
        print(f"    - Detected Tree Count: {m.tree_count}")
        print(f"    - Raw Bounding Box Area: {m.raw_box_area_m2:.1f} m2")
        print(f"    - Dissolved Box Area: {m.dissolved_box_area_m2:.1f} m2")
        print(f"    - Dissolved Canopy Area: {m.dissolved_canopy_area_m2:.1f} m2 ({m.dissolved_canopy_area_ha:.4f} ha)")
        print(f"    - Canopy Cover: {m.canopy_cover_percentage:.1f}%")
        print(f"    - Mean Crown Diameter: {m.mean_crown_diameter_m:.2f} m")
        print(f"    - Overlap Redundancy Ratio: {m.overlap_redundancy_pct:.1f}%")

        # Scientific integrity assertions
        assert m.tree_count == len(detections), "Pipeline tree count should match detector count"
        assert m.dissolved_canopy_area_m2 > 0.0, "Canopy area must be strictly positive"
        assert m.raw_box_area_m2 >= m.dissolved_box_area_m2, (
            "Raw box area must be >= dissolved box area (overlaps must be removed)"
        )
        assert m.dissolved_box_area_m2 >= m.dissolved_canopy_area_m2, (
            "Dissolved rectangular box area must be >= dissolved elliptical crown area"
        )

        # Check export files
        exported = pipeline_result.exported_files
        print(f"    - Exported GIS files: {list(exported.keys())}")
        for fmt, fpath in exported.items():
            assert os.path.exists(fpath), f"Expected export file {fpath} does not exist"
            assert os.path.getsize(fpath) > 0, f"Export file {fpath} is empty"

    # 8. Complete Real Geospatial Workflow with Georeferenced GeoTIFF + KML AOI Clipping
    print("\n[*] Executing Full Geospatial Workflow: Real GeoTIFF (UTM 17N) + KML Clipping...")
    with tempfile.TemporaryDirectory() as geo_tmpdir:
        import rasterio
        from rasterio.transform import from_origin
        from src.geometry import reproject_geometry
        from shapely.geometry import box

        # Create real georeferenced GeoTIFF: 0.10 m/px, origin (403500, 3285000), CRS=EPSG:32617
        tif_path = os.path.join(geo_tmpdir, "osbs_georeferenced.tif")
        transform = from_origin(403500.0, 3285000.0, 0.10, 0.10)
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            width=w,
            height=h,
            count=3,
            dtype="uint8",
            crs="EPSG:32617",
            transform=transform
        ) as ds:
            ds.write(np.transpose(img_rgb, (2, 0, 1)))

        # Create KML boundary polygon covering 25x25m section in the center
        kml_box_utm = box(403508.0, 3284968.0, 403532.0, 3284992.0)
        kml_box_wgs84 = reproject_geometry(kml_box_utm, "EPSG:32617", "EPSG:4326")
        coords_str = " ".join(f"{x},{y},0" for x, y in kml_box_wgs84.exterior.coords)

        kml_path = os.path.join(geo_tmpdir, "forest_boundary.kml")
        with open(kml_path, "w", encoding="utf-8") as f:
            f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
            <kml xmlns="http://www.opengis.net/kml/2.2">
              <Document><Placemark><name>Forest AOI Stand</name><Polygon><outerBoundaryIs><LinearRing>
                <coordinates>{coords_str}</coordinates>
              </LinearRing></outerBoundaryIs></Polygon></Placemark></Document>
            </kml>""")

        t_geo_start = time.perf_counter()
        geo_result = run_forest_crown_pipeline(
            image_source=tif_path,
            user_gsd=None,  # Strict GeoTIFF metadata provenance
            aoi_source=kml_path,
            confidence_threshold=0.20,
            patch_size=400,
            patch_overlap=0.15,
            export_dir=geo_tmpdir,
            export_base_name="kml_clipped_forest",
            detector=detector,
            use_mock_detector=False
        )
        t_geo_end = time.perf_counter()

        gm = geo_result.metrics
        print(f"[OK] Real GeoTIFF + KML workflow completed in {t_geo_end - t_geo_start:.3f} seconds.")
        print(f"    - Tree Count: {gm.tree_count}")
        print(f"    - Forest AOI Extent: {gm.forest_area_m2:.1f} m2 ({gm.forest_area_ha:.4f} ha)")
        print(f"    - Unique Canopy Area: {gm.unique_canopy_area_m2:.1f} m2")
        print(f"    - Canopy Cover: {gm.canopy_cover_pct:.1f}%")
        print(f"    - Detected Crown Area: {gm.detected_crown_area_m2:.1f} m2")
        print(f"    - Mean Crown Area: {gm.mean_crown_area_m2:.2f} m2")
        print(f"    - Median Crown Area: {gm.median_crown_area_m2:.2f} m2")
        print(f"    - Median Confidence: {gm.median_detection_confidence:.4f}")
        print(f"    - Low Confidence Count: {gm.low_confidence_count}")
        print(f"    - Overlap Redundancy: {gm.overlap_redundancy_pct:.1f}%")
        print(f"    - Resolution (GSD X/Y): {gm.gsd_x:.4f} / {gm.gsd_y:.4f} m/px")
        print(f"    - CRS: {gm.crs}")
        print(f"    - Crown Geometry Method: {gm.crown_geometry_method}")
        print(f"    - Quality Flags: {gm.quality_flags}")

        # Verification of Requirement H & I
        assert gm.tree_count > 0, "Expected trees detected inside clipped AOI"
        assert gm.unique_canopy_area_m2 > 0.0, "Unique canopy area must be > 0"
        assert gm.forest_area_m2 is not None and gm.forest_area_m2 > 0.0, "Forest AOI area must be > 0"
        assert gm.canopy_cover_pct is not None, "Canopy cover percentage must be computed"
        assert gm.crown_geometry_method == "inscribed_ellipse_from_detection_bbox"
        assert gm.crs == "EPSG:32617", "CRS must match GeoTIFF projected UTM coordinate system"
        assert len(gm.quality_flags) >= 2, "Expected quality flags present"

        # Check export files
        for ext in [".geojson", ".csv"]:
            found = any(f.endswith(ext) for f in geo_result.exported_files.values())
            assert found, f"Expected {ext} file exported"

    print("\n" + "=" * 70)
    print("SUCCESS: SMOKE TEST PASSED: REAL GEOSPATIAL WORKFLOW VERIFIED END-TO-END")
    print("=" * 70)
    return {
        "python_version": sys.version.split()[0],
        "deepforest_version": deepforest.__version__,
        "init_time_sec": init_duration,
        "inference_time_sec": inf_duration,
        "detections_count": len(detections),
    }


if __name__ == "__main__":
    run_smoke_test()
