"""
End-to-end integration tests for pipeline.py.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from pipeline import run_forest_crown_pipeline
from src.detection import MockTreeDetector
from src.io_utils import MissingGSDError

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from shapely.geometry import box
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_raises_on_missing_gsd(self):
        # 200x200 image array
        synthetic_img = np.full((200, 200, 3), 120, dtype=np.uint8)

        # Without user_gsd, pipeline must fail
        with self.assertRaises(MissingGSDError):
            run_forest_crown_pipeline(
                image_source=synthetic_img,
                user_gsd=None,
                use_mock_detector=True
            )

    def test_pipeline_runs_with_user_gsd_and_mock_detector(self):
        # 300x300 image with green patches
        synthetic_img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Add forest green background
        synthetic_img[:, :] = (34, 139, 34)

        result = run_forest_crown_pipeline(
            image_source=synthetic_img,
            user_gsd=0.08,
            confidence_threshold=0.20,
            patch_size=200,
            patch_overlap=0.10,
            export_dir=self.temp_dir,
            export_base_name="test_run",
            use_mock_detector=True
        )

        # Verify results
        self.assertGreater(result.metrics.tree_count, 0)
        self.assertGreater(result.metrics.dissolved_canopy_area_m2, 0.0)
        self.assertEqual(result.metrics.gsd_source, "user_specified")
        self.assertAlmostEqual(result.metrics.gsd_m, 0.08)
        self.assertIn("LOCAL_METRIC_CARTESIAN", result.metrics.metric_crs)

        # Check export files were generated
        if HAS_SHAPELY:
            self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_run_crowns.csv")))


if __name__ == "__main__":
    unittest.main()
