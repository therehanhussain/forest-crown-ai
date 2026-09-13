"""
Tree crown detection module with DeepForest integration and Non-Maximum Suppression (NMS).

Enforces:
- Clean modular interface separating model loading from inference.
- Reliable NMS for stitching sliding-window detections on large orthomosaics.
- Graceful test fallback for environments without pre-downloaded PyTorch weights.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

try:
    import deepforest
    from deepforest import main as df_main
    HAS_DEEPFOREST = True
except ImportError:
    deepforest = None
    df_main = None
    HAS_DEEPFOREST = False


@dataclass
class DetectionBox:
    """Represents a single detected tree crown bounding box."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    label: str = "Tree"
    geometry_geo: Optional[Any] = None
    detection_id: Optional[int] = None

    @property
    def width_pixels(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height_pixels(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area_pixels(self) -> float:
        return self.width_pixels * self.height_pixels

    @property
    def center_pixel(self) -> Tuple[float, float]:
        return (self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0

    @property
    def approx_diameter_pixels(self) -> float:
        """Equivalent diameter from bounding box dimensions."""
        return (self.width_pixels + self.height_pixels) / 2.0

    def approx_area_m2(self, gsd_m: float) -> float:
        """Inscribed ellipse area in square meters using GSD."""
        w_m = self.width_pixels * gsd_m
        h_m = self.height_pixels * gsd_m
        return math.pi * (w_m / 2.0) * (h_m / 2.0)


def compute_iou(box_a: DetectionBox, box_b: DetectionBox) -> float:
    """Computes Intersection-over-Union (IoU) between two detection boxes."""
    x_left = max(box_a.xmin, box_b.xmin)
    y_top = max(box_a.ymin, box_b.ymin)
    x_right = min(box_a.xmax, box_b.xmax)
    y_bottom = min(box_a.ymax, box_b.ymax)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = box_a.area_pixels + box_b.area_pixels - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def apply_non_max_suppression(
    boxes: List[DetectionBox],
    iou_threshold: float = 0.3
) -> List[DetectionBox]:
    """
    Applies standard Non-Maximum Suppression (NMS) to eliminate duplicate bounding boxes
    generated across overlapping sliding-window tiles.
    """
    if not boxes:
        return []

    # Sort boxes by confidence score descending
    sorted_boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept_boxes = []

    while sorted_boxes:
        best_box = sorted_boxes.pop(0)
        kept_boxes.append(best_box)

        # Filter out remaining boxes that have IoU > iou_threshold with best_box
        sorted_boxes = [
            b for b in sorted_boxes
            if compute_iou(best_box, b) < iou_threshold
        ]

    return kept_boxes


def translate_detection_box(box: DetectionBox, col_offset: float, row_offset: float) -> DetectionBox:
    """Translates a tile-local detection box to global image coordinates."""
    return DetectionBox(
        xmin=box.xmin + col_offset,
        ymin=box.ymin + row_offset,
        xmax=box.xmax + col_offset,
        ymax=box.ymax + row_offset,
        confidence=box.confidence,
        label=box.label,
        geometry_geo=box.geometry_geo
    )


def predict_with_tiling(
    detector: BaseTreeDetector,
    image_rgb: np.ndarray,
    patch_size: int = 400,
    patch_overlap: float = 0.15,
    score_threshold: float = 0.15,
    iou_threshold: float = 0.30
) -> List[DetectionBox]:
    """
    Executes windowed sliding-tile inference across an image:
    1. Slices image into overlapping tiles using patch_size and patch_overlap.
    2. Runs detector on each tile.
    3. Translates tile-local detections back to global image pixel coordinates.
    4. Applies Non-Maximum Suppression (NMS) across tile boundaries to eliminate duplicate detections.
    """
    h, w = image_rgb.shape[:2]
    step = int(patch_size * (1.0 - patch_overlap))
    step = max(1, step)

    all_detections: List[DetectionBox] = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            actual_w = min(patch_size, w - x)
            actual_h = min(patch_size, h - y)

            tile = image_rgb[y:y + actual_h, x:x + actual_w]
            tile_detections = detector.predict(
                image_rgb=tile,
                score_threshold=score_threshold,
                patch_size=patch_size,
                patch_overlap=0.0
            )

            for det in tile_detections:
                translated = translate_detection_box(det, col_offset=float(x), row_offset=float(y))
                all_detections.append(translated)

    # Eliminate duplicate detections at tile boundaries
    return apply_non_max_suppression(all_detections, iou_threshold=iou_threshold)


class BaseTreeDetector(ABC):
    """Abstract interface for tree crown detection models."""

    @abstractmethod
    def load_model(self) -> None:
        """Loads model weights."""
        pass

    @abstractmethod
    def predict(
        self,
        image_rgb: np.ndarray,
        score_threshold: float = 0.15,
        patch_size: int = 400,
        patch_overlap: float = 0.15
    ) -> List[DetectionBox]:
        """Runs detection on an RGB image array."""
        pass


def load_deepforest_model(model_name: Optional[str] = None) -> Any:
    """
    Encapsulates DeepForest neural network initialization and release weights loading.
    Designed for clean separation of model loading from inference, and seamless
    integration with caching mechanisms such as Streamlit's `@st.cache_resource`.

    In DeepForest 2.x, default initialization automatically loads and checks the release model.
    """
    if not HAS_DEEPFOREST:
        raise ImportError(
            "DeepForest is not installed. "
            "Please run `pip install deepforest` to enable neural tree crown detection."
        )
    model = df_main.deepforest()
    if model_name:
        model.load_model(model_name=model_name)
    return model


class DeepForestDetector(BaseTreeDetector):
    """Production implementation using DeepForest (RetinaNet on ResNet50)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_instance: Optional[Any] = None
    ):
        self.model_path = model_path
        self.model = model_instance
        self._is_loaded = (model_instance is not None)

    def load_model(self) -> None:
        """Loads or attaches the DeepForest model instance."""
        if not HAS_DEEPFOREST:
            raise ImportError(
                "DeepForest is not installed. "
                "Please run `pip install deepforest` to enable neural tree crown detection."
            )
        if self.model is None:
            self.model = load_deepforest_model(model_name=self.model_path)
            self._is_loaded = True

    def predict(
        self,
        image_rgb: np.ndarray,
        score_threshold: float = 0.15,
        patch_size: int = 400,
        patch_overlap: float = 0.15
    ) -> List[DetectionBox]:
        """
        Runs individual tree crown detection on an RGB image array using DeepForest 2.x.
        Handles both small chips (predict_image) and large orthomosaics (predict_tile).
        """
        if not self._is_loaded or self.model is None:
            self.load_model()

        # DeepForest expects float32 or uint8; passing float32 avoids conversion warnings
        if image_rgb.dtype == np.uint8:
            image_input = image_rgb.astype(np.float32)
        else:
            image_input = image_rgb

        h, w = image_input.shape[:2]

        # Use predict_tile for large rasters, or predict_image for single chips
        if w > patch_size or h > patch_size:
            df_predictions = self.model.predict_tile(
                image=image_input,
                patch_size=patch_size,
                patch_overlap=patch_overlap
            )
        else:
            df_predictions = self.model.predict_image(
                image=image_input
            )

        detections = []
        if df_predictions is not None and not df_predictions.empty:
            for _, row in df_predictions.iterrows():
                # Defensively accept both 'score' and 'confidence'
                score = float(row.get("score", row.get("confidence", 0.0)))
                if score >= score_threshold:
                    detections.append(DetectionBox(
                        xmin=float(row["xmin"]),
                        ymin=float(row["ymin"]),
                        xmax=float(row["xmax"]),
                        ymax=float(row["ymax"]),
                        confidence=score,
                        label=str(row.get("label", "Tree"))
                    ))

        return detections


class MockTreeDetector(BaseTreeDetector):
    """
    Lightweight, deterministic detector for unit tests and headless verification
    where heavy PyTorch models are unavailable or inappropriate.
    """

    def __init__(self, synthetic_grid_step: int = 60, crown_radius: int = 15):
        self.synthetic_grid_step = synthetic_grid_step
        self.crown_radius = crown_radius
        self._is_loaded = True

    def load_model(self) -> None:
        self._is_loaded = True

    def predict(
        self,
        image_rgb: np.ndarray,
        score_threshold: float = 0.15,
        patch_size: int = 400,
        patch_overlap: float = 0.15
    ) -> List[DetectionBox]:
        h, w = image_rgb.shape[:2]
        boxes = []
        step = self.synthetic_grid_step
        r = self.crown_radius

        for y in range(r + 10, h - r - 10, step):
            for x in range(r + 10, w - r - 10, step):
                # Calculate simple green excess if pixels exist: 2*G - R - B
                patch = image_rgb[y - r:y + r, x - r:x + r]
                if patch.size > 0:
                    r_val = float(np.mean(patch[:, :, 0]))
                    g_val = float(np.mean(patch[:, :, 1]))
                    b_val = float(np.mean(patch[:, :, 2]))
                    greenness = (2.0 * g_val - r_val - b_val)
                    # Baseline confidence
                    conf = min(0.95, max(0.50, 0.50 + greenness / 500.0))
                else:
                    conf = 0.85

                if conf >= score_threshold:
                    boxes.append(DetectionBox(
                        xmin=float(x - r),
                        ymin=float(y - r),
                        xmax=float(x + r),
                        ymax=float(y + r),
                        confidence=conf,
                        label="Tree"
                    ))

        return boxes
