"""
Comprehensive automated tests for the complete real geospatial workflow in Forest Crown AI.

Covers all 11 scientific & geospatial requirements specified in Requirement J:
1. GeoTIFF GSD extraction (X and Y resolution from affine transform).
2. JPG with missing GSD -> expected failure (MissingGSDError).
3. KML/image overlap (Polygon, MultiPolygon, interior holes).
4. KML/image no-overlap -> expected failure (GeospatialError).
5. AOI clipping (raster cropping, affine preservation, outside masking).
6. Pixel-to-world conversion (UTM metric coordinates).
7. Crown area calculation (inscribed ellipses in m^2).
8. Overlapping crown union (spatial union dissolution without double-counting).
9. Tile coordinate translation (local window -> global image).
10. Duplicate detection removal (NMS across overlapping tile boundaries).
11. Real DeepForest inference on georeferenced GeoTIFF with KML clipping.
"""

from __future__ import annotations

import io
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline import run_forest_crown_pipeline
from src.detection import (
    DeepForestDetector,
    DetectionBox,
    MockTreeDetector,
    apply_non_max_suppression,
    compute_iou,
    load_deepforest_model,
    predict_with_tiling,
    translate_detection_box,
)
from src.geometry import (
    OverlapStatus,
    check_aoi_image_overlap,
    dissolve_geometries,
    estimate_utm_epsg,
    get_metric_projected_crs,
    pixel_box_to_geo_polygon,
    pixel_box_to_metric_polygon,
    reproject_geometry,
)
from src.io_utils import (
    GeospatialError,
    MissingGSDError,
    load_kml_or_kmz,
    read_geotiff_metadata,
)
from src.metrics import calculate_canopy_metrics
from src.preprocessing import clip_raster_to_polygon

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from shapely.geometry import MultiPolygon, Point, Polygon, box
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import deepforest
    HAS_DEEPFOREST = True
except ImportError:
    HAS_DEEPFOREST = False


class TestGeospatialWorkflow(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. GeoTIFF GSD extraction
    # -------------------------------------------------------------------------
    @unittest.skipUnless(HAS_RASTERIO, "Rasterio required for GeoTIFF test")
    def test_geotiff_gsd_extraction(self):
        tif_path = os.path.join(self.temp_dir, "test_res.tif")
        # 0.08 m/px in X, 0.12 m/px in Y
        transform = from_origin(400000.0, 3200000.0, 0.08, 0.12)
        arr = np.zeros((3, 100, 100), dtype=np.uint8)

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            width=100,
            height=100,
            count=3,
            dtype="uint8",
            crs="EPSG:32617",
            transform=transform
        ) as ds:
            ds.write(arr)

        meta = read_geotiff_metadata(tif_path, user_gsd=None)
        self.assertTrue(meta.has_georeference)
        self.assertAlmostEqual(meta.gsd_x_m, 0.08, places=4)
        self.assertAlmostEqual(meta.gsd_y_m, 0.12, places=4)
        self.assertEqual(meta.gsd_source, "metadata_projected")
        self.assertIn("32617", meta.crs)

        # Ensure user_gsd NEVER overrides embedded GeoTIFF metadata
        meta_with_override = read_geotiff_metadata(tif_path, user_gsd=0.50)
        self.assertAlmostEqual(meta_with_override.gsd_x_m, 0.08, places=4)
        self.assertAlmostEqual(meta_with_override.gsd_y_m, 0.12, places=4)
        self.assertEqual(meta_with_override.gsd_source, "metadata_projected")

    # -------------------------------------------------------------------------
    # 2. JPG with missing GSD -> expected failure
    # -------------------------------------------------------------------------
    @unittest.skipUnless(HAS_PIL, "PIL required for image test")
    def test_jpg_missing_gsd_expected_failure(self):
        jpg_path = os.path.join(self.temp_dir, "forest.jpg")
        img = Image.new("RGB", (150, 150), (34, 139, 34))
        img.save(jpg_path, "JPEG")

        with self.assertRaises(MissingGSDError):
            read_geotiff_metadata(jpg_path, user_gsd=None)

        with self.assertRaises(MissingGSDError):
            run_forest_crown_pipeline(
                image_source=jpg_path,
                user_gsd=None,
                use_mock_detector=True
            )

    # -------------------------------------------------------------------------
    # 3. KML/image overlap (Polygon, MultiPolygon, interior holes)
    # -------------------------------------------------------------------------
    @unittest.skipUnless(HAS_SHAPELY, "Shapely required for overlap tests")
    def test_kml_image_overlap_polygon_and_holes(self):
        # Image bounds: 400,000 to 400,100 m (100x100m extent in UTM Zone 17N)
        img_bounds = (400000.0, 3200000.0, 400100.0, 3200100.0)
        img_crs = "EPSG:32617"

        # Polygon with interior hole
        exterior = [(400020, 3200020), (400080, 3200020), (400080, 3200080), (400020, 3200080), (400020, 3200020)]
        hole = [(400040, 3200040), (400060, 3200040), (400060, 3200060), (400040, 3200060), (400040, 3200040)]
        poly_with_hole = Polygon(exterior, holes=[hole])

        result = check_aoi_image_overlap(img_bounds, img_crs, poly_with_hole, img_crs)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.status, OverlapStatus.FULL_INSIDE)
        # Exterior area 60x60 = 3600, hole 20x20 = 400 -> net area 3200 m^2
        self.assertAlmostEqual(result.aoi_area_m2, 3200.0, delta=1.0)
        self.assertAlmostEqual(result.pct_aoi_covered, 100.0, delta=0.1)

        # MultiPolygon test
        p1 = box(400010, 3200010, 400040, 3200040)
        p2 = box(400060, 3200060, 400090, 3200090)
        multipoly = MultiPolygon([p1, p2])
        res_multi = check_aoi_image_overlap(img_bounds, img_crs, multipoly, img_crs)
        self.assertTrue(res_multi.is_valid)
        self.assertEqual(res_multi.status, OverlapStatus.FULL_INSIDE)

    # -------------------------------------------------------------------------
    # 4. KML/image no-overlap -> expected failure
    # -------------------------------------------------------------------------
    @unittest.skipUnless(HAS_SHAPELY, "Shapely required for no-overlap tests")
    def test_kml_image_no_overlap_expected_failure(self):
        # Florida raster (UTM Zone 17N)
        img_bounds = (400000.0, 3200000.0, 400100.0, 3200100.0)
        img_crs = "EPSG:32617"

        # California AOI far away
        california_aoi = box(-122.5, 37.7, -122.4, 37.8)

        overlap_res = check_aoi_image_overlap(img_bounds, img_crs, california_aoi, "EPSG:4326")
        self.assertFalse(overlap_res.is_valid)
        self.assertEqual(overlap_res.status, OverlapStatus.NO_OVERLAP)

        # Test pipeline halts with GeospatialError
        if HAS_RASTERIO:
            tif_path = os.path.join(self.temp_dir, "florida.tif")
            transform = from_origin(400000.0, 3200100.0, 0.10, 0.10)
            arr = np.zeros((3, 50, 50), dtype=np.uint8)
            with rasterio.open(tif_path, "w", driver="GTiff", width=50, height=50, count=3, dtype="uint8", crs=img_crs, transform=transform) as ds:
                ds.write(arr)

            kml_content = """<?xml version="1.0" encoding="UTF-8"?>
            <kml xmlns="http://www.opengis.net/kml/2.2">
              <Document><Placemark><Polygon><outerBoundaryIs><LinearRing>
                <coordinates>-122.5,37.7,0 -122.4,37.7,0 -122.4,37.8,0 -122.5,37.8,0 -122.5,37.7,0</coordinates>
              </LinearRing></outerBoundaryIs></Polygon></Placemark></Document>
            </kml>""".encode("utf-8")

            with self.assertRaises(GeospatialError):
                run_forest_crown_pipeline(
                    image_source=tif_path,
                    aoi_source=kml_content,
                    use_mock_detector=True
                )

    # -------------------------------------------------------------------------
    # 5. AOI clipping (cropping, affine preservation, outside masking)
    # -------------------------------------------------------------------------
    @unittest.skipUnless(HAS_SHAPELY, "Shapely required for clipping tests")
    def test_aoi_clipping_and_affine_preservation(self):
        # 100x100 pixel image, origin (1000, 2000), 1 m/px
        img_arr = np.full((100, 100, 3), 200, dtype=np.uint8)
        transform = (1.0, 0.0, 1000.0, 0.0, -1.0, 2000.0)

        # AOI from pixel (20, 20) to (60, 60): world coords x=[1020, 1060], y=[1940, 1980]
        aoi_poly = box(1020.0, 1940.0, 1060.0, 1980.0)

        cropped, new_transform, offset_px = clip_raster_to_polygon(
            img_arr=img_arr,
            transform=transform,
            polygon_geo=aoi_poly
        )

        # Check cropped size: 40x40 pixels
        self.assertEqual(cropped.shape[0], 40)
        self.assertEqual(cropped.shape[1], 40)
        self.assertEqual(offset_px, (20, 20))

        # Check preserved affine transform
        # new_c = 1000 + 20*1 = 1020; new_f = 2000 + 20*(-1) = 1980
        self.assertAlmostEqual(new_transform[2], 1020.0)
        self.assertAlmostEqual(new_transform[5], 1980.0)

        # Test triangle polygon to verify outside masking
        triangle = Polygon([(1020, 1940), (1060, 1940), (1020, 1980), (1020, 1940)])
        cropped_tri, _, _ = clip_raster_to_polygon(img_arr, transform, triangle)
        # Top-right corner (row 0, col 39) is outside the triangle and should be masked to 0
        self.assertEqual(int(cropped_tri[0, 39, 0]), 0)

    # -------------------------------------------------------------------------
    # 6. Pixel-to-world conversion
    # -------------------------------------------------------------------------
    @unittest.skipUnless(HAS_SHAPELY, "Shapely required")
    def test_pixel_to_world_conversion(self):
        # 10 m/px transform: x_geo = 500000 + 10 * x_pix; y_geo = 4000000 - 10 * y_pix
        transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
        box_coords = (10.0, 20.0, 30.0, 40.0)

        geo_box = pixel_box_to_geo_polygon(box_coords, transform, as_ellipse=False)
        self.assertIsNotNone(geo_box)
        minx, miny, maxx, maxy = geo_box.bounds

        # xmin: 500000 + 10*10 = 500100; xmax: 500000 + 10*30 = 500300
        self.assertAlmostEqual(minx, 500100.0)
        self.assertAlmostEqual(maxx, 500300.0)
        # ymin: 4000000 - 10*40 = 3999600; ymax: 4000000 - 10*20 = 3999800
        self.assertAlmostEqual(miny, 3999600.0)
        self.assertAlmostEqual(maxy, 3999800.0)

    # -------------------------------------------------------------------------
    # 7. Crown area calculation
    # -------------------------------------------------------------------------
    @unittest.skipUnless(HAS_SHAPELY, "Shapely required")
    def test_crown_area_calculation(self):
        # Box 20x20 m -> rectangle area = 400 m^2
        # Inscribed ellipse: r = 10 -> area = pi * 10^2 = ~314.16 m^2
        rect_poly = pixel_box_to_metric_polygon((0, 0, 20, 20), gsd_m=1.0, as_ellipse=False)
        ellipse_poly = pixel_box_to_metric_polygon((0, 0, 20, 20), gsd_m=1.0, as_ellipse=True, num_pts=32)

        self.assertAlmostEqual(rect_poly.area, 400.0)
        expected_ellipse = math.pi * 100.0
        self.assertAlmostEqual(ellipse_poly.area, expected_ellipse, delta=3.0)
        ratio = ellipse_poly.area / rect_poly.area
        self.assertAlmostEqual(ratio, math.pi / 4.0, delta=0.01)

    # -------------------------------------------------------------------------
    # 8. Overlapping crown union (dissolution without double-counting)
    # -------------------------------------------------------------------------
    @unittest.skipUnless(HAS_SHAPELY, "Shapely required")
    def test_overlapping_crown_union(self):
        # Crown 1: center (10, 10), radius 5 -> area ~78.5
        # Crown 2: center (15, 10), radius 5 -> area ~78.5
        c1 = Point(10, 10).buffer(5.0)
        c2 = Point(15, 10).buffer(5.0)

        dissolved, dissolved_area, raw_sum = dissolve_geometries([c1, c2])
        self.assertAlmostEqual(raw_sum, c1.area + c2.area)
        self.assertLess(dissolved_area, raw_sum)
        # Redundancy strictly positive
        redundancy_pct = (raw_sum - dissolved_area) / raw_sum * 100.0
        self.assertGreater(redundancy_pct, 10.0)

    # -------------------------------------------------------------------------
    # 9. Tile coordinate translation
    # -------------------------------------------------------------------------
    def test_tile_coordinate_translation(self):
        local_box = DetectionBox(xmin=10.0, ymin=15.0, xmax=50.0, ymax=60.0, confidence=0.88)
        col_off = 400.0
        row_off = 350.0

        global_box = translate_detection_box(local_box, col_off, row_off)
        self.assertEqual(global_box.xmin, 410.0)
        self.assertEqual(global_box.ymin, 365.0)
        self.assertEqual(global_box.xmax, 450.0)
        self.assertEqual(global_box.ymax, 410.0)
        self.assertEqual(global_box.confidence, 0.88)

    # -------------------------------------------------------------------------
    # 10. Duplicate detection removal across tile boundaries
    # -------------------------------------------------------------------------
    def test_duplicate_detection_removal(self):
        # Two boxes detecting the same tree on adjacent tiles with slight offset
        box_tile1 = DetectionBox(xmin=395.0, ymin=100.0, xmax=425.0, ymax=130.0, confidence=0.82)
        box_tile2 = DetectionBox(xmin=396.0, ymin=101.0, xmax=426.0, ymax=131.0, confidence=0.89)
        different_tree = DetectionBox(xmin=50.0, ymin=50.0, xmax=80.0, ymax=80.0, confidence=0.91)

        # Before NMS: 3 boxes
        boxes = [box_tile1, box_tile2, different_tree]
        kept = apply_non_max_suppression(boxes, iou_threshold=0.30)

        # After NMS: 2 boxes (the higher confidence duplicate kept, lower discarded)
        self.assertEqual(len(kept), 2)
        confidences = [b.confidence for b in kept]
        self.assertIn(0.89, confidences)
        self.assertIn(0.91, confidences)
        self.assertNotIn(0.82, confidences)

    # -------------------------------------------------------------------------
    # 11. Real DeepForest inference on georeferenced GeoTIFF with KML clipping
    # -------------------------------------------------------------------------
    @unittest.skipUnless(HAS_DEEPFOREST and HAS_RASTERIO and HAS_SHAPELY, "DeepForest, Rasterio and Shapely required")
    def test_real_deepforest_inference_geotiff_kml_workflow(self):
        sample_img_path = deepforest.get_data("OSBS_029.png")
        pil_img = Image.open(sample_img_path)
        img_arr = np.array(pil_img.convert("RGB"))
        h, w, _ = img_arr.shape

        # Create real georeferenced GeoTIFF in UTM Zone 17N (Ordway-Swisher, Florida)
        # GSD = 0.10 m/pixel
        tif_path = os.path.join(self.temp_dir, "osbs_real.tif")
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
            ds.write(np.transpose(img_arr, (2, 0, 1)))

        # Create KML boundary covering center 20x20 meters of the 40x40m raster
        # Top-left in UTM: (403500, 3285000)
        # Center in UTM: (403510, 3284970) to (403530, 3284990)
        aoi_poly_utm = box(403510.0, 3284970.0, 403530.0, 3284990.0)
        # Reproject to WGS84 for KML
        aoi_wgs84 = reproject_geometry(aoi_poly_utm, "EPSG:32617", "EPSG:4326")
        coords_str = " ".join(f"{x},{y},0" for x, y in aoi_wgs84.exterior.coords)

        kml_path = os.path.join(self.temp_dir, "aoi_center.kml")
        kml_text = f"""<?xml version="1.0" encoding="UTF-8"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document><Placemark><name>Forest AOI</name><Polygon><outerBoundaryIs><LinearRing>
            <coordinates>{coords_str}</coordinates>
          </LinearRing></outerBoundaryIs></Polygon></Placemark></Document>
        </kml>"""
        with open(kml_path, "w", encoding="utf-8") as f:
            f.write(kml_text)

        # Initialize real DeepForest model
        model = load_deepforest_model()
        detector = DeepForestDetector(model_instance=model)

        # Run pipeline
        result = run_forest_crown_pipeline(
            image_source=tif_path,
            user_gsd=None, # Strict GeoTIFF GSD provenance
            aoi_source=kml_path,
            confidence_threshold=0.20,
            patch_size=400,
            patch_overlap=0.15,
            export_dir=self.temp_dir,
            export_base_name="real_test",
            detector=detector,
            use_mock_detector=False
        )

        m = result.metrics
        self.assertGreater(m.tree_count, 0)
        self.assertGreater(m.unique_canopy_area_m2, 0.0)
        self.assertIsNotNone(m.forest_area_m2)
        self.assertGreater(m.forest_area_m2, 0.0)
        self.assertIsNotNone(m.canopy_cover_pct)
        self.assertEqual(m.crown_geometry_method, "inscribed_ellipse_from_detection_bbox")
        self.assertEqual(m.crs, "EPSG:32617")
        self.assertAlmostEqual(m.gsd_x, 0.10, places=3)
        self.assertAlmostEqual(m.gsd_y, 0.10, places=3)

        # Check export files
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "real_test_crowns.geojson")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "real_test_crowns.csv")))


if __name__ == "__main__":
    unittest.main()
