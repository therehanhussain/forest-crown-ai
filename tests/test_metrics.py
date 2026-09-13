"""
Unit tests for src/metrics.py.
"""

import unittest

from src.metrics import calculate_canopy_metrics

try:
    from shapely.geometry import box
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


class TestMetrics(unittest.TestCase):

    @unittest.skipUnless(HAS_SHAPELY, "Shapely is required for metric tests")
    def test_metrics_separation_and_dissolution(self):
        # Create 4 tree boxes that partially overlap
        # 10x10 meters each (raw area = 100 m^2 each -> raw sum = 400 m^2)
        box1 = box(0, 0, 10, 10)
        box2 = box(5, 0, 15, 10)  # Overlaps box1 by 50 m^2
        box3 = box(20, 20, 30, 30)
        box4 = box(25, 20, 35, 30) # Overlaps box3 by 50 m^2

        crown_metric_boxes = [box1, box2, box3, box4]
        # Inscribed crown ellipses (simulated here as slightly smaller buffered shapes)
        crown_metric_polygons = [b.buffer(-0.5) for b in crown_metric_boxes]

        crown_diameters = [10.0, 10.0, 10.0, 10.0]
        crown_confidences = [0.85, 0.90, 0.75, 0.80]

        aoi_poly = box(0, 0, 100, 100) # 10,000 m^2 = 1.0 hectare

        metrics = calculate_canopy_metrics(
            crown_diameters_m=crown_diameters,
            crown_metric_polygons=crown_metric_polygons,
            crown_metric_boxes=crown_metric_boxes,
            crown_confidences=crown_confidences,
            aoi_metric_polygon=aoi_poly,
            gsd_m=0.10,
            gsd_source="metadata_projected",
            metric_crs="EPSG:32610",
            confidence_threshold=0.20
        )

        # 1. Verify tree count is separate from area
        self.assertEqual(metrics.tree_count, 4)

        # 2. Raw box area should be 400 m^2
        self.assertAlmostEqual(metrics.raw_box_area_m2, 400.0)

        # 3. Dissolved box area should be 300 m^2 (since 2 pairs each overlap by 50 m^2)
        self.assertAlmostEqual(metrics.dissolved_box_area_m2, 300.0)
        self.assertAlmostEqual(metrics.overlap_redundancy_m2, 100.0)
        self.assertAlmostEqual(metrics.overlap_redundancy_pct, 25.0)

        # 4. Stand density & canopy cover
        self.assertEqual(metrics.aoi_total_area_ha, 1.0)
        self.assertAlmostEqual(metrics.trees_per_hectare, 4.0)
        self.assertGreater(metrics.canopy_cover_percentage, 0.0)
        self.assertLess(metrics.canopy_cover_percentage, 10.0)

        # 5. Ethical carbon credit disclaimer check
        self.assertIn("DISCLAIMER", metrics.carbon_disclaimer)
        self.assertIn("carbon", metrics.carbon_disclaimer.lower())
        self.assertGreater(len(metrics.assumptions), 0)
        self.assertGreater(len(metrics.limitations), 0)

    def test_zero_detections_handled_cleanly(self):
        metrics = calculate_canopy_metrics(
            crown_diameters_m=[],
            crown_metric_polygons=[],
            crown_metric_boxes=[],
            crown_confidences=[],
            aoi_metric_polygon=None,
            gsd_m=0.10,
            gsd_source="user_specified",
            metric_crs="LOCAL_METRIC_CARTESIAN (meters)",
            confidence_threshold=0.20
        )

        self.assertEqual(metrics.tree_count, 0)
        self.assertEqual(metrics.dissolved_canopy_area_m2, 0.0)
        self.assertIsNotNone(metrics.carbon_disclaimer)


if __name__ == "__main__":
    unittest.main()
