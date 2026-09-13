"""
Unit tests for src/geometry.py.
"""

import math
import unittest

from src.geometry import (
    OverlapStatus,
    check_aoi_image_overlap,
    dissolve_geometries,
    estimate_utm_epsg,
    is_crs_geographic,
    pixel_box_to_metric_polygon,
)

try:
    from shapely.geometry import Polygon, box
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


class TestGeometry(unittest.TestCase):

    def test_utm_zone_estimation(self):
        # San Francisco (-122.4, 37.8) -> UTM Zone 10 North -> EPSG:32610
        epsg_sf = estimate_utm_epsg(-122.4, 37.8)
        self.assertEqual(epsg_sf, 32610)

        # Paris (2.35, 48.85) -> UTM Zone 31 North -> EPSG:32631
        epsg_paris = estimate_utm_epsg(2.35, 48.85)
        self.assertEqual(epsg_paris, 32631)

        # Sydney (151.2, -33.86) -> UTM Zone 56 South -> EPSG:32756
        epsg_sydney = estimate_utm_epsg(151.2, -33.86)
        self.assertEqual(epsg_sydney, 32756)

    def test_crs_geographic_detection(self):
        self.assertTrue(is_crs_geographic("EPSG:4326"))
        self.assertTrue(is_crs_geographic("urn:ogc:def:crs:OGC:1.3:CRS84"))
        self.assertTrue(is_crs_geographic("WGS 84"))
        self.assertFalse(is_crs_geographic("EPSG:32610"))

    @unittest.skipUnless(HAS_SHAPELY, "Shapely is required for geometry tests")
    def test_dissolve_geometries_removes_double_counting(self):
        # Two 10x10 squares that overlap by 50%
        # Box 1: (0, 0) to (10, 10), area = 100
        # Box 2: (5, 0) to (15, 10), area = 100
        # Total naive sum = 200
        # Dissolved union = (0, 0) to (15, 10), area = 150
        box1 = box(0, 0, 10, 10)
        box2 = box(5, 0, 15, 10)

        dissolved_geom, dissolved_area, raw_sum = dissolve_geometries([box1, box2])

        self.assertAlmostEqual(raw_sum, 200.0)
        self.assertAlmostEqual(dissolved_area, 150.0)
        # Redundancy is 50.0
        redundancy = raw_sum - dissolved_area
        self.assertAlmostEqual(redundancy, 50.0)
        self.assertLess(dissolved_area, raw_sum)

    @unittest.skipUnless(HAS_SHAPELY, "Shapely is required for geometry tests")
    def test_inscribed_ellipse_crown_reduction(self):
        # Rectangular box 10x10 -> area 100 m^2
        # Inscribed ellipse: r=5 -> area = pi * 5^2 = ~78.54 m^2
        # Ratio should be ~pi/4 (~0.785)
        rect_poly = pixel_box_to_metric_polygon((0, 0, 10, 10), gsd_m=1.0, as_ellipse=False)
        ellipse_poly = pixel_box_to_metric_polygon((0, 0, 10, 10), gsd_m=1.0, as_ellipse=True, num_pts=32)

        self.assertAlmostEqual(rect_poly.area, 100.0)
        expected_ellipse_area = math.pi * 25.0
        self.assertAlmostEqual(ellipse_poly.area, expected_ellipse_area, delta=1.0)
        self.assertLess(ellipse_poly.area, rect_poly.area)

    @unittest.skipUnless(HAS_SHAPELY, "Shapely is required for geometry tests")
    def test_check_aoi_image_overlap_no_overlap(self):
        # Image bounds in UTM meters
        img_bounds = (500000, 4000000, 501000, 4001000)
        img_crs = "EPSG:32610"

        # Distant AOI far away
        distant_aoi = box(600000, 5000000, 601000, 5001000)
        result = check_aoi_image_overlap(img_bounds, img_crs, distant_aoi, "EPSG:32610")

        self.assertEqual(result.status, OverlapStatus.NO_OVERLAP)
        self.assertFalse(result.is_valid)
        self.assertIsNotNone(result.warning_message)

    @unittest.skipUnless(HAS_SHAPELY, "Shapely is required for geometry tests")
    def test_check_aoi_image_overlap_full_inside(self):
        img_bounds = (500000, 4000000, 502000, 4002000)
        img_crs = "EPSG:32610"

        # AOI inside image
        inner_aoi = box(500500, 4000500, 501500, 4001500)
        result = check_aoi_image_overlap(img_bounds, img_crs, inner_aoi, "EPSG:32610")

        self.assertEqual(result.status, OverlapStatus.FULL_INSIDE)
        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.pct_aoi_covered, 100.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
