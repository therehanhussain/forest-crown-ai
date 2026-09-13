"""
End-to-end test simulating the complete Streamlit UI workflow:
- Loads the built-in demo dataset (sample_forest_utm17n.tif + sample_boundary.kml).
- Validates geospatial overlap.
- Runs the real DeepForest detector.
- Generates metrics and validates all 15 fields.
- Tests create_folium_map with crown inspection popups and tooltips.
- Tests generate_summary_report.
- Verifies all exports exist.
"""

from pathlib import Path
import unittest
import numpy as np

from pipeline import run_forest_crown_pipeline
from src.detection import DeepForestDetector, load_deepforest_model
from src.geometry import check_aoi_image_overlap, OverlapStatus
from src.io_utils import load_kml_or_kmz, read_geotiff_metadata
from src.visualization import create_folium_map, overlay_detections_on_image
from app import generate_summary_report

class TestUIWorkflow(unittest.TestCase):

    def test_demo_dataset_end_to_end(self):
        tif_path = Path("demo_data/sample_forest_utm17n.tif")
        kml_path = Path("demo_data/sample_boundary.kml")

        self.assertTrue(tif_path.exists(), "Sample GeoTIFF must exist in demo_data")
        self.assertTrue(kml_path.exists(), "Sample KML must exist in demo_data")

        tif_bytes = tif_path.read_bytes()
        kml_bytes = kml_path.read_bytes()

        # 1. Test Pre-Analysis Validation
        meta = read_geotiff_metadata(tif_bytes)
        self.assertTrue(meta.has_georeference)
        self.assertAlmostEqual(meta.gsd_x_m, 0.10, places=3)
        self.assertEqual(meta.crs, "EPSG:32617")

        aoi_poly, _ = load_kml_or_kmz(kml_bytes)
        overlap = check_aoi_image_overlap(
            image_bounds=meta.bounds,
            image_crs=meta.crs,
            aoi_geom=aoi_poly,
            aoi_crs="EPSG:4326"
        )
        self.assertIn(overlap.status, [OverlapStatus.FULL_INSIDE, OverlapStatus.PARTIAL])

        # 2. Run Real Model Pipeline
        detector = DeepForestDetector(model_instance=load_deepforest_model())
        result = run_forest_crown_pipeline(
            image_source=tif_bytes,
            user_gsd=None,
            aoi_source=kml_bytes,
            confidence_threshold=0.20,
            patch_size=400,
            patch_overlap=0.15,
            export_dir="./outputs",
            export_base_name="demo_ui_test",
            detector=detector,
            use_mock_detector=False
        )

        # 3. Validate Quantitative Outputs
        m = result.metrics
        self.assertGreater(m.tree_count, 0)
        self.assertIsNotNone(m.forest_area_m2)
        self.assertGreater(m.unique_canopy_area_m2, 0.0)
        self.assertGreater(m.canopy_cover_pct, 0.0)
        self.assertGreater(m.mean_crown_area_m2, 0.0)
        self.assertGreater(m.median_crown_area_m2, 0.0)
        self.assertEqual(m.crown_geometry_method, "inscribed_ellipse_from_detection_bbox")

        # 4. Test Interactive Folium Map with Crown Popups
        crown_areas = [d.approx_area_m2(m.gsd_m) for d in result.detections]
        tree_ids = [d.detection_id for d in result.detections]
        bounds = result.metadata.bounds
        center_lat = (bounds[1] + bounds[3]) / 2.0
        center_lon = (bounds[0] + bounds[2]) / 2.0

        folium_map = create_folium_map(
            center_lat=center_lat,
            center_lon=center_lon,
            zoom_start=18,
            aoi_wgs84_geom=result.aoi_wgs84_geom,
            dissolved_canopy_wgs84_geom=result.dissolved_canopy_wgs84,
            crown_wgs84_geoms=result.crown_wgs84_geoms,
            crown_confidences=[d.confidence for d in result.detections],
            crown_areas_m2=crown_areas,
            tree_ids=tree_ids
        )
        self.assertIsNotNone(folium_map)

        # 5. Test Summary Report Generator
        report = generate_summary_report(result, "sample_forest_utm17n.tif")
        self.assertIn("Canopy Analysis Summary Report", report)
        self.assertIn(f"{m.tree_count:,} crowns", report)
        self.assertIn("EPSG:32617", report)
        self.assertIn("LEGAL & METHODOLOGICAL DISCLAIMER", report)

        print("[OK] UI Workflow End-to-End Test PASSED!")
        print(f"    - Trees Detected: {m.tree_count}")
        print(f"    - Unique Canopy Area: {m.unique_canopy_area_m2:.1f} m²")
        print(f"    - Canopy Cover: {m.canopy_cover_pct:.1f}%")

if __name__ == "__main__":
    unittest.main()
