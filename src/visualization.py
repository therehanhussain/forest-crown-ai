"""
Visualization components for Forest Crown AI:
- Overlay detection boxes on RGB imagery using PIL.
- Interactive dual-layer geospatial maps with Folium (Satellite Basemap, AOI, Dissolved Canopy).
- Plotly charts for crown size distributions and overlap redundancy comparison.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None
    HAS_PIL = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    go = None
    HAS_PLOTLY = False

try:
    import folium
    from folium import plugins
    HAS_FOLIUM = True
except ImportError:
    folium = None
    plugins = None
    HAS_FOLIUM = False

try:
    from shapely.geometry import mapping
    HAS_SHAPELY = True
except ImportError:
    mapping = None
    HAS_SHAPELY = False


def overlay_detections_on_image(
    image_rgb: np.ndarray,
    detections: List[Any],
    line_width: int = 2,
    draw_center: bool = True
) -> Any:
    """
    Renders bounding boxes and crown markers onto an RGB image.
    Uses confidence-based color gradients (Green for high, Orange for medium, Yellow for low).
    """
    if not HAS_PIL:
        return image_rgb

    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img)

    for det in detections:
        xmin = getattr(det, 'xmin', 0.0)
        ymin = getattr(det, 'ymin', 0.0)
        xmax = getattr(det, 'xmax', 0.0)
        ymax = getattr(det, 'ymax', 0.0)
        conf = getattr(det, 'confidence', 1.0)

        # Color based on confidence
        if conf >= 0.70:
            box_color = (0, 255, 127)      # Spring green
        elif conf >= 0.40:
            box_color = (255, 215, 0)      # Gold
        else:
            box_color = (255, 99, 71)       # Tomato red

        # Draw rectangle
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=box_color, width=line_width)

        # Draw center point
        if draw_center:
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            r = max(2, line_width)
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=box_color)

    return img


def create_folium_map(
    center_lat: float,
    center_lon: float,
    zoom_start: int = 17,
    aoi_wgs84_geom: Optional[Any] = None,
    dissolved_canopy_wgs84_geom: Optional[Any] = None,
    crown_wgs84_geoms: Optional[List[Any]] = None,
    crown_confidences: Optional[List[float]] = None,
    crown_areas_m2: Optional[List[float]] = None,
    tree_ids: Optional[List[int]] = None
) -> Optional[Any]:
    """
    Generates an interactive Folium map featuring:
    - High-resolution Esri World Imagery Satellite basemap
    - User AOI boundary layer
    - Dissolved canopy cover footprint layer
    - Individual tree crown markers/polygons layer with inspection popups & tooltips
    """
    if not HAS_FOLIUM or not HAS_SHAPELY:
        return None

    # Base map with Esri World Imagery
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        max_zoom=22,
        tiles=None
    )

    # Esri Satellite
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri, Maxar, Earthstar Geographics, USDA, USGS',
        name='Satellite (Esri World Imagery)',
        overlay=False,
        control=True
    ).add_to(m)

    # Standard OpenStreetMap
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OpenStreetMap',
        overlay=False,
        control=True
    ).add_to(m)

    # 1. Add AOI Boundary Layer
    if aoi_wgs84_geom is not None and not aoi_wgs84_geom.is_empty:
        aoi_layer = folium.FeatureGroup(name="AOI Boundary", show=True)
        folium.GeoJson(
            mapping(aoi_wgs84_geom),
            style_function=lambda x: {
                'fillColor': '#00FFFF',
                'color': '#00B0FF',
                'weight': 3,
                'fillOpacity': 0.12,
                'dashArray': '6, 6'
            },
            tooltip="Forest Area of Interest (AOI)"
        ).add_to(aoi_layer)
        aoi_layer.add_to(m)

    # 2. Add Dissolved Canopy Footprint Layer
    if dissolved_canopy_wgs84_geom is not None and not dissolved_canopy_wgs84_geom.is_empty:
        canopy_layer = folium.FeatureGroup(name="Dissolved Canopy Cover", show=True)
        folium.GeoJson(
            mapping(dissolved_canopy_wgs84_geom),
            style_function=lambda x: {
                'fillColor': '#2E7D32',
                'color': '#1B5E20',
                'weight': 1,
                'fillOpacity': 0.45
            },
            tooltip="Dissolved Canopy Cover (Overlap Removed)"
        ).add_to(canopy_layer)
        canopy_layer.add_to(m)

    # 3. Add Individual Tree Crowns with Inspection Tooltips and Popups
    if crown_wgs84_geoms:
        trees_layer = folium.FeatureGroup(name=f"Individual Crowns ({len(crown_wgs84_geoms)})", show=True)
        conf_list = crown_confidences or [1.0] * len(crown_wgs84_geoms)
        areas_list = crown_areas_m2 or [0.0] * len(crown_wgs84_geoms)
        ids_list = tree_ids or [i + 1 for i in range(len(crown_wgs84_geoms))]

        # Add up to 500 crown geometries directly to avoid browser slowdown
        for i, g in enumerate(crown_wgs84_geoms[:500]):
            if g and not g.is_empty:
                conf = conf_list[i] if i < len(conf_list) else 1.0
                tid = ids_list[i] if i < len(ids_list) else (i + 1)
                area_val = areas_list[i] if i < len(areas_list) else 0.0
                area_str = f"{area_val:.2f} m²" if area_val > 0 else "Approximated"

                popup_html = (
                    f"<div style='font-family: sans-serif; font-size: 13px; line-height: 1.4; min-width: 140px;'>"
                    f"<h4 style='margin: 0 0 5px 0; color: #1B5E20;'>Tree #{tid}</h4>"
                    f"<b>Confidence:</b> {conf:.1%}<br>"
                    f"<b>Crown Area:</b> {area_str}<br>"
                    f"<small style='color: #666;'>Inscribed Ellipse Approx.</small>"
                    f"</div>"
                )
                tooltip_str = f"Tree #{tid} | Conf: {conf:.1%} | Area: {area_str}"

                folium.GeoJson(
                    mapping(g),
                    style_function=lambda x, c=conf: {
                        'fillColor': '#76FF03' if c >= 0.50 else '#FFD600',
                        'color': '#33691E' if c >= 0.50 else '#E65100',
                        'weight': 1.5,
                        'fillOpacity': 0.4
                    },
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=tooltip_str
                ).add_to(trees_layer)
        trees_layer.add_to(m)

    # Clean professional GIS legend
    legend_html = '''
    <div style="
        position: fixed; 
        bottom: 25px; left: 25px; width: 180px; height: auto;
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.1);
        z-index: 9999;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 11px;
        color: #1e293b;
        padding: 9px 12px;
        line-height: 1.4;
    ">
        <div style="font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: #0f172a; margin-bottom: 6px; font-size: 10px;">Layers</div>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <span style="display: inline-block; width: 12px; height: 12px; background-color: rgba(0, 176, 255, 0.25); border: 2px dashed #00B0FF; margin-right: 8px; border-radius: 2px;"></span>
            <span>AOI Boundary</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <span style="display: inline-block; width: 12px; height: 12px; background-color: rgba(46, 125, 50, 0.5); border: 1px solid #1B5E20; margin-right: 8px; border-radius: 2px;"></span>
            <span>Dissolved Canopy</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <span style="display: inline-block; width: 10px; height: 10px; background-color: #76FF03; border: 1.5px solid #33691E; margin-right: 9px; border-radius: 50%;"></span>
            <span>Crown (≥ 0.50)</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="display: inline-block; width: 10px; height: 10px; background-color: #FFD600; border: 1.5px solid #E65100; margin-right: 9px; border-radius: 50%;"></span>
            <span>Crown (< 0.50)</span>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def plot_crown_diameter_distribution(diameters_m: List[float]) -> Optional[Any]:
    """Generates an interactive Plotly histogram of crown diameters in meters."""
    if not HAS_PLOTLY or not diameters_m:
        return None

    mean_d = float(np.mean(diameters_m))
    median_d = float(np.median(diameters_m))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=diameters_m,
        nbinsx=25,
        marker_color='#2E7D32',
        opacity=0.75,
        name='Crowns'
    ))

    fig.add_vline(x=mean_d, line_dash="dash", line_color="#D32F2F", annotation_text=f"Mean: {mean_d:.1f}m")
    fig.add_vline(x=median_d, line_dash="dot", line_color="#F57C00", annotation_text=f"Median: {median_d:.1f}m")

    fig.update_layout(
        title="Tree Crown Diameter Distribution (Meters)",
        xaxis_title="Estimated Crown Diameter (m)",
        yaxis_title="Count",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_area_comparison_chart(
    raw_box_area_m2: float,
    dissolved_box_area_m2: float,
    dissolved_canopy_area_m2: float
) -> Optional[Any]:
    """
    Plots a direct comparison between:
    1. Naive Bounding Box Sum (Overestimation)
    2. Dissolved Bounding Boxes (Removes overlap double-counting)
    3. Realistic Dissolved Canopy (Elliptical crowns)
    """
    if not HAS_PLOTLY:
        return None

    categories = [
        "1. Naive Box Sum<br>(Overestimated)",
        "2. Dissolved Box Union<br>(Overlap Removed)",
        "3. Realistic Crown Union<br>(Inscribed Ellipses)"
    ]
    values = [raw_box_area_m2, dissolved_box_area_m2, dissolved_canopy_area_m2]
    colors = ["#E53935", "#FB8C00", "#43A047"]

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        text=[f"{v:,.1f} m²" for v in values],
        textposition="auto",
        marker_color=colors
    ))

    fig.update_layout(
        title="Canopy Area Estimation Comparison (m²)",
        yaxis_title="Area (m²)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
