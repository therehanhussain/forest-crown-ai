"""
Unit tests for src/io_utils.py.
"""

import io
import unittest
import zipfile
from pathlib import Path

from src.io_utils import (
    GeospatialError,
    MissingGSDError,
    compute_metric_gsd_from_geographic,
    load_kml_or_kmz,
    parse_kml_coordinates,
    read_geotiff_metadata,
)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from shapely.geometry import Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


class TestIOUtils(unittest.TestCase):

    def test_parse_kml_coordinates(self):
        coord_text = """
            -122.1430,37.4419,0
            -122.1400,37.4419,0
            -122.1400,37.4400,0
            -122.1430,37.4400,0
            -122.1430,37.4419,0
        """
        pts = parse_kml_coordinates(coord_text)
        self.assertEqual(len(pts), 5)
        self.assertAlmostEqual(pts[0][0], -122.1430)
        self.assertAlmostEqual(pts[0][1], 37.4419)

    def test_compute_metric_gsd_from_geographic(self):
        # 1 deg at equator is approx 111,320 m
        deg_res = 0.0001
        gsd_x, gsd_y = compute_metric_gsd_from_geographic(deg_res, deg_res, center_lat_deg=0.0)
        self.assertGreater(gsd_x, 10.0)
        self.assertLess(gsd_x, 12.0)
        self.assertGreater(gsd_y, 10.0)
        self.assertLess(gsd_y, 12.0)

    @unittest.skipUnless(HAS_PIL, "PIL is required for image creation")
    def test_missing_gsd_raises_error(self):
        # Create small test PNG image in memory
        img = Image.new("RGB", (100, 100), color=(34, 139, 34))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        # Calling read_geotiff_metadata on PNG without user_gsd MUST raise MissingGSDError
        with self.assertRaises(MissingGSDError):
            read_geotiff_metadata(buf, user_gsd=None)

    @unittest.skipUnless(HAS_PIL, "PIL is required for image creation")
    def test_user_gsd_accepted_for_standard_image(self):
        img = Image.new("RGB", (200, 150), color=(34, 139, 34))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        meta = read_geotiff_metadata(buf, user_gsd=0.05)
        self.assertEqual(meta.width, 200)
        self.assertEqual(meta.height, 150)
        self.assertAlmostEqual(meta.gsd_x_m, 0.05)
        self.assertEqual(meta.gsd_source, "user_specified")
        self.assertFalse(meta.has_georeference)

    def test_load_kml_pure_xml(self):
        sample_kml = """<?xml version="1.0" encoding="UTF-8"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document>
            <Placemark>
              <name>Test Forest AOI</name>
              <Polygon>
                <outerBoundaryIs>
                  <LinearRing>
                    <coordinates>
                      -122.14,37.44,0 -122.13,37.44,0 -122.13,37.43,0 -122.14,37.43,0 -122.14,37.44,0
                    </coordinates>
                  </LinearRing>
                </outerBoundaryIs>
              </Polygon>
            </Placemark>
          </Document>
        </kml>
        """.encode('utf-8')

        geom, crs = load_kml_or_kmz(sample_kml)
        self.assertEqual(crs, "EPSG:4326")
        if HAS_SHAPELY:
            self.assertTrue(geom.is_valid)
            self.assertEqual(geom.geom_type, "Polygon")

    def test_load_kmz_archive(self):
        sample_kml = """<?xml version="1.0" encoding="UTF-8"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document>
            <Placemark>
              <name>KMZ Forest Stand</name>
              <Polygon>
                <outerBoundaryIs>
                  <LinearRing>
                    <coordinates>
                      10.0,50.0,0 10.1,50.0,0 10.1,49.9,0 10.0,49.9,0 10.0,50.0,0
                    </coordinates>
                  </LinearRing>
                </outerBoundaryIs>
              </Polygon>
            </Placemark>
          </Document>
        </kml>
        """.encode('utf-8')

        # Zip it into in-memory KMZ
        kmz_buf = io.BytesIO()
        with zipfile.ZipFile(kmz_buf, 'w') as zf:
            zf.writestr('doc.kml', sample_kml)
        kmz_buf.seek(0)

        geom, crs = load_kml_or_kmz(kmz_buf)
        self.assertEqual(crs, "EPSG:4326")
        if HAS_SHAPELY:
            self.assertTrue(geom.is_valid)


if __name__ == "__main__":
    unittest.main()
