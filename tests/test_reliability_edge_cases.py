"""
Automated edge-case and reliability testing for Forest Crown AI.
Tests the 8 mandatory failure modes and edge scenarios:
1. Demo dataset -> Analyze
2. GeoTIFF + KML upload -> Analyze
3. Missing CRS (standard imagery) with user GSD
4. Missing GSD for JPG/PNG (must raise MissingGSDError)
5. KML with no overlap (must raise GeospatialError)
6. Invalid/corrupt file bytes (must raise GeospatialError/InvalidAOIError)
7. Extreme confidence thresholds (low=0.10, high=0.95)
8. Empty detection result handling (0 detections safety)
"""

from pathlib import Path
import unittest
import numpy as np
from shapely.geometry import box

from pipeline import run_forest_crown_pipeline
from src.detection import DeepForestDetector, MockTreeDetector, load_deepforest_model
from src.geometry import check_aoi_image_overlap, OverlapStatus
from src.io_utils import (
    GeospatialError,
    InvalidAOIError,
    MissingGSDError,
    load_kml_or_kmz,
    read_geotiff_metadata,
)
from src.metrics import calculate_canopy_metrics

class TestReliabilityEdgeCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Cache real detector for tests
        cls.detector = DeepForestDetector(model_instance=load_deepforest_model())
        cls.tif_bytes = Path("demo_data/sample_forest_utm17n.tif").read_bytes()
        cls.kml_bytes = Path("demo_data/sample_boundary.kml").read_bytes()
        cls.png_bytes = Path("demo_data/sample_aerial_photo.png").read_bytes()

    def test_case_1_demo_dataset_analyze(self):
        """Scenario 1: Demo dataset execution."""
        res = run_forest_crown_pipeline(
            self.tif_bytes,
            aoi_source=self.kml_bytes,
            detector=self.detector,
            use_mock_detector=False
        )
        self.assertGreater(res.metrics.tree_count, 0)
        self.assertEqual(res.metrics.crs, "EPSG:32617")
        self.assertEqual(res.metrics.gsd_source, "metadata_projected")

    def test_case_2_geotiff_kml_upload_analyze(self):
        """Scenario 2: GeoTIFF + KML workflow."""
        meta = read_geotiff_metadata(self.tif_bytes)
        poly, _ = load_kml_or_kmz(self.kml_bytes)
        overlap = check_aoi_image_overlap(meta.bounds, meta.crs, poly, "EPSG:4326")
        self.assertIn(overlap.status, [OverlapStatus.FULL_INSIDE, OverlapStatus.PARTIAL])

        res = run_forest_crown_pipeline(
            self.tif_bytes,
            aoi_source=self.kml_bytes,
            detector=self.detector
        )
        self.assertIsNotNone(res.metrics.forest_area_m2)
        self.assertGreater(res.metrics.canopy_cover_pct, 0.0)

    def test_case_3_missing_crs_with_user_gsd(self):
        """Scenario 3: Standard PNG image without CRS runs in local metric coordinates."""
        res = run_forest_crown_pipeline(
            self.png_bytes,
            user_gsd=0.10,
            detector=self.detector
        )
        self.assertGreater(res.metrics.tree_count, 0)
        self.assertEqual(res.metrics.gsd_source, "user_specified")
        self.assertIn("LOCAL_METRIC", res.metrics.crs)
        self.assertIn("User-supplied GSD", res.metrics.quality_flags)

    def test_case_4_missing_gsd_raises_clean_error(self):
        """Scenario 4: Missing GSD for standard image raises MissingGSDError."""
        with self.assertRaises(MissingGSDError) as ctx:
            run_forest_crown_pipeline(
                self.png_bytes,
                user_gsd=None,
                detector=self.detector
            )
        self.assertIn("GSD", str(ctx.exception))

    def test_case_5_kml_no_overlap_raises_clean_error(self):
        """Scenario 5: KML boundary outside image bounds raises GeospatialError."""
        # Non-overlapping KML in California while GeoTIFF is in Florida
        california_kml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>
            -122.0,37.0,0 -122.0,37.1,0 -121.9,37.1,0 -121.9,37.0,0 -122.0,37.0,0
          </coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>
        </kml>"""

        with self.assertRaises(GeospatialError) as ctx:
            run_forest_crown_pipeline(
                self.tif_bytes,
                aoi_source=california_kml,
                detector=self.detector
            )
        self.assertIn("does not intersect", str(ctx.exception))

    def test_case_6_corrupt_file_handling(self):
        """Scenario 6: Corrupt file bytes produce clean error without unhandled crash."""
        corrupt_bytes = b"CORRUPT_NOT_A_VALID_IMAGE_OR_TIFF"
        with self.assertRaises(GeospatialError):
            read_geotiff_metadata(corrupt_bytes)

        with self.assertRaises(InvalidAOIError):
            load_kml_or_kmz(b"CORRUPT_NOT_A_KML")

    def test_case_7_extreme_confidence_thresholds(self):
        """Scenario 7: Low (0.10) and high (0.95) thresholds execute smoothly."""
        res_low = run_forest_crown_pipeline(
            self.tif_bytes,
            aoi_source=self.kml_bytes,
            confidence_threshold=0.10,
            detector=self.detector
        )
        self.assertGreater(res_low.metrics.tree_count, 0)

        res_high = run_forest_crown_pipeline(
            self.tif_bytes,
            aoi_source=self.kml_bytes,
            confidence_threshold=0.95,
            detector=self.detector
        )
        self.assertGreaterEqual(res_high.metrics.tree_count, 0)
        self.assertLessEqual(res_high.metrics.tree_count, res_low.metrics.tree_count)

    def test_case_8_empty_detection_result_handling(self):
        """Scenario 8: 0 detections handled gracefully without division by zero."""
        empty_metrics = calculate_canopy_metrics(
            crown_diameters_m=[],
            crown_metric_polygons=[],
            crown_metric_boxes=[],
            crown_confidences=[],
            aoi_metric_polygon=box(0, 0, 100, 100),
            gsd_m=0.10,
            gsd_source="user_specified",
            metric_crs="EPSG:32617",
            confidence_threshold=0.99
        )
        self.assertEqual(empty_metrics.tree_count, 0)
        self.assertEqual(empty_metrics.detected_crown_area_m2, 0.0)
        self.assertEqual(empty_metrics.unique_canopy_area_m2, 0.0)
        self.assertEqual(empty_metrics.canopy_cover_pct, 0.0)
        self.assertEqual(empty_metrics.overlap_redundancy_pct, 0.0)
        self.assertIn("No tree crowns detected", empty_metrics.limitations[0])

if __name__ == "__main__":
    unittest.main()
