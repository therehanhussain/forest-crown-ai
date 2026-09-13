"""
Forest Crown AI — Presentation Layer & Self-Service Web Application.

Professional commercial carbon-tech and geospatial intelligence platform for:
- Autonomous Individual Tree Crown Detection (DeepForest 2.1.0)
- True Continuous Canopy Area Estimation (Inscribed Elliptical Dissolve)
- Non-overlapping Stand Density and Canopy Coverage Analytics
- Standard GIS Interoperability (GeoJSON, Shapefile, CSV, Google Earth KML)
"""

from __future__ import annotations

import io
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    go = None
    HAS_PLOTLY = False

from pipeline import PipelineResult, run_forest_crown_pipeline
from src.geometry import OverlapStatus, check_aoi_image_overlap
from src.io_utils import (
    GeospatialError,
    MissingGSDError,
    load_kml_or_kmz,
    read_geotiff_metadata,
)
from src.visualization import (
    create_folium_map,
    overlay_detections_on_image,
)

# Streamlit Page Configuration — 100% Full Width, No Sidebar
st.set_page_config(
    page_title="Forest Crown AI | Canopy Intelligence",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Carbon-Tech Design System (CSS)
st.markdown(
    """
    <style>
    /* Complete Sidebar Elimination — Zero Chevron / Drawer / Hamburger */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"],
    div[data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapseButton"],
    div[data-testid="stSidebarNav"] {
        display: none !important;
        visibility: hidden !important;
        width: 0px !important;
        height: 0px !important;
        pointer-events: none !important;
    }

    /* Global Reset & Dark Slate Canvas */
    html, body, [class*="css"], .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        background-color: #0B0F19 !important;
        color: #F1F5F9;
    }
    header[data-testid="stHeader"] {
        background-color: #0B0F19 !important;
    }
    .block-container {
        max-width: 1440px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Hero Section */
    .hero-container {
        padding: 2px 0 10px 0;
        margin-bottom: 6px;
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #F8FAFC;
        line-height: 1.15;
        margin-bottom: 3px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        font-weight: 500;
        color: #94A3B8;
        margin-bottom: 4px;
        line-height: 1.35;
    }
    .hero-tagline {
        font-size: 0.84rem;
        font-style: italic;
        color: #64748B;
        line-height: 1.35;
    }

    /* Elegant Workflow Stepper */
    .workflow-bar {
        display: inline-flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 6px;
        padding: 6px 14px;
        margin-bottom: 18px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.06em;
    }
    .workflow-step {
        color: #CBD5E1;
    }
    .workflow-arrow {
        color: #4B5563;
        font-size: 0.70rem;
    }

    /* Primary Analysis Input Card */
    .analysis-input-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .input-card-title {
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .input-card-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-bottom: 14px;
    }

    /* Upload Cards */
    .upload-card-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .upload-card-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #334155;
        margin-bottom: 8px;
    }

    /* Benchmark Description Badge */
    .benchmark-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 9px 14px;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 6px;
        font-size: 0.82rem;
        color: #065F46;
        font-weight: 500;
        margin-top: 4px;
        margin-bottom: 12px;
    }
    .benchmark-badge-tag {
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #047857;
    }

    /* Widget Labels & Controls */
    div[data-testid="stWidgetLabel"] label p,
    div[data-testid="stWidgetLabel"] label span {
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: #94A3B8 !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
        background-color: #059669 !important;
        border-color: #047857 !important;
    }

    /* Primary Centered Action Button */
    div.stButton > button[kind="primary"] {
        background-color: #059669 !important;
        color: #FFFFFF !important;
        border: 1px solid #10B981 !important;
        border-radius: 6px !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.08em !important;
        padding: 0.65rem 1.6rem !important;
        min-height: 48px !important;
        box-shadow: 0 2px 6px rgba(5, 150, 105, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
        text-transform: uppercase !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #047857 !important;
        border-color: #059669 !important;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.4) !important;
        color: #FFFFFF !important;
    }

    /* Secondary & Export Buttons */
    div.stDownloadButton > button {
        background-color: #1E293B !important;
        color: #F1F5F9 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        padding: 0.50rem 1rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stDownloadButton > button:hover {
        border-color: #10B981 !important;
        color: #A7F3D0 !important;
        background-color: #064E3B !important;
    }

    /* Input Validation Card */
    .validation-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 10px 16px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }
    .val-header {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 6px;
    }
    .val-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 8px;
        align-items: center;
    }
    @media (max-width: 960px) {
        .val-grid {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    .val-col {
        display: flex;
        flex-direction: column;
    }
    .val-lbl {
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 2px;
    }
    .val-txt {
        font-size: 0.82rem;
        font-weight: 600;
        color: #0F172A;
    }
    .val-badge-ready {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #065F46;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        padding: 3px 6px;
        border-radius: 4px;
        text-align: center;
        white-space: nowrap;
        display: inline-block;
    }
    .val-badge-blocked {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #991B1B;
        background: #FEF2F2;
        border: 1px solid #FECACA;
        padding: 3px 6px;
        border-radius: 4px;
        text-align: center;
        white-space: nowrap;
        display: inline-block;
    }

    /* Ready Callout (Before Analysis) */
    .ready-callout {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .callout-badge {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        color: #166534;
        background: #DCFCE7;
        padding: 4px 10px;
        border-radius: 4px;
        white-space: nowrap;
    }
    .callout-content {
        display: flex;
        flex-direction: column;
    }
    .callout-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #14532D;
    }
    .callout-sub {
        font-size: 0.80rem;
        color: #15803D;
        margin-top: 2px;
    }

    /* Six Responsive KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 12px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    @media (max-width: 1350px) {
        .kpi-container {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    @media (max-width: 768px) {
        .kpi-container {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 4px;
        white-space: normal;
        line-height: 1.25;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.15;
        font-feature-settings: 'tnum' on, 'lnum' on;
    }
    .kpi-sub {
        font-size: 0.74rem;
        font-weight: 600;
        color: #059669;
        margin-top: 3px;
    }

    /* Section Headings */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #F8FAFC;
        margin-top: 20px;
        margin-bottom: 2px;
    }
    .section-sub {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-bottom: 12px;
    }

    /* Quality Card */
    .quality-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 14px 18px;
        height: 100%;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }
    .quality-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #F1F5F9;
        font-size: 0.82rem;
    }
    .quality-row:last-child {
        border-bottom: none;
    }
    .quality-key {
        font-weight: 600;
        color: #64748B;
    }
    .quality-val {
        font-weight: 600;
        color: #0F172A;
    }

    /* Model Limitations Card */
    .limitations-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 14px 18px;
        height: 100%;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        font-size: 0.82rem;
        color: #334155;
        line-height: 1.45;
    }
    .limitations-item {
        margin-bottom: 6px;
        padding-left: 12px;
        text-indent: -12px;
    }
    .limitations-bullet {
        color: #DC2626;
        font-weight: 700;
        margin-right: 4px;
    }

    /* Methodology Grid */
    .method-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 10px;
        margin-top: 8px;
        margin-bottom: 20px;
    }
    .method-step {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px 14px;
        font-size: 0.82rem;
        line-height: 1.4;
        color: #334155;
    }
    .method-num {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #059669;
        margin-bottom: 3px;
        text-transform: uppercase;
    }

    /* Footer */
    .app-footer {
        margin-top: 40px;
        padding-top: 18px;
        border-top: 1px solid #1E293B;
        font-size: 0.78rem;
        color: #64748B;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource(show_spinner="Initializing DeepForest neural network weights...")
def get_cached_deepforest_detector():
    """Caches the heavy PyTorch DeepForest model across Streamlit sessions."""
    from src.detection import DeepForestDetector, load_deepforest_model
    model = load_deepforest_model()
    return DeepForestDetector(model_instance=model)


def plot_crown_size_distribution_dark(crown_areas_m2: List[float]):
    """Generates an interactive Plotly histogram of crown areas matching the carbon-tech dark theme."""
    if not HAS_PLOTLY or not crown_areas_m2:
        return None
    mean_a = float(np.mean(crown_areas_m2))
    median_a = float(np.median(crown_areas_m2))
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=crown_areas_m2,
        nbinsx=25,
        marker_color='#10B981',
        marker_line_color='#047857',
        marker_line_width=1,
        opacity=0.85,
        name='Crowns'
    ))
    fig.add_vline(x=mean_a, line_dash="dash", line_color="#34D399", annotation_text=f"Mean: {mean_a:.1f} m²", annotation_font_color="#A7F3D0")
    fig.add_vline(x=median_a, line_dash="dot", line_color="#FBBF24", annotation_text=f"Median: {median_a:.1f} m²", annotation_font_color="#FDE68A")
    fig.update_layout(
        title=dict(text="Crown Size Distribution (m²)", font=dict(color="#F1F5F9", size=13)),
        xaxis=dict(title="Crown Area (m²)", color="#94A3B8", gridcolor="#1E293B", zerolinecolor="#374151"),
        yaxis=dict(title="Tree Count", color="#94A3B8", gridcolor="#1E293B", zerolinecolor="#374151"),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(l=40, r=30, t=40, b=40),
        height=320
    )
    return fig


def plot_detection_confidence_dark(confidences: List[float]):
    """Generates an interactive Plotly histogram of detection confidence scores."""
    if not HAS_PLOTLY or not confidences:
        return None
    median_c = float(np.median(confidences))
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=confidences,
        nbinsx=20,
        marker_color='#059669',
        marker_line_color='#047857',
        marker_line_width=1,
        opacity=0.85,
        name='Confidence'
    ))
    fig.add_vline(x=median_c, line_dash="dash", line_color="#6EE7B7", annotation_text=f"Median: {median_c:.2f}", annotation_font_color="#A7F3D0")
    fig.update_layout(
        title=dict(text="Detection Confidence Distribution", font=dict(color="#F1F5F9", size=13)),
        xaxis=dict(title="Confidence Score", color="#94A3B8", gridcolor="#1E293B", zerolinecolor="#374151", range=[0.0, 1.0]),
        yaxis=dict(title="Tree Count", color="#94A3B8", gridcolor="#1E293B", zerolinecolor="#374151"),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(l=40, r=30, t=40, b=40),
        height=320
    )
    return fig


def generate_summary_report(result: PipelineResult, image_filename: str) -> str:
    """Generates a structured markdown analysis summary report."""
    m = result.metrics
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Forest Crown AI — Canopy Analysis Summary Report",
        f"**Generated:** {timestamp}",
        f"**Source Imagery:** `{image_filename}`",
        f"**Coordinate Reference System:** `{m.crs}`",
        f"**Spatial Resolution (GSD):** X = {m.gsd_x:.4f} m/px, Y = {m.gsd_y:.4f} m/px (Source: {m.gsd_source})",
        f"**Crown Geometry Method:** `{m.crown_geometry_method}`",
        "",
        "## Key Quantitative Results",
        f"- **Detected Tree Count:** {m.tree_count:,} crowns",
        f"- **Forest Area of Interest (AOI):** {m.forest_area_m2:,.1f} m² ({m.forest_area_ha:.4f} ha)" if m.forest_area_m2 else "- **Forest Area:** Full image extent",
        f"- **Unique Canopy Area (Dissolved):** {m.unique_canopy_area_m2:,.1f} m² ({m.dissolved_canopy_area_ha:.4f} ha)",
        f"- **Raw Bounding Box Area Sum:** {m.raw_box_area_m2:,.1f} m²",
        f"- **Canopy Cover Percentage:** {m.canopy_cover_pct:.1f}%" if m.canopy_cover_pct is not None else "- **Canopy Cover:** N/A",
        f"- **Overlap Redundancy Ratio:** {m.overlap_redundancy_pct:.1f}% ({m.overlap_redundancy_m2:,.1f} m² shared/overlapping)",
        f"- **Mean Crown Area:** {m.mean_crown_area_m2:.2f} m²",
        f"- **Median Crown Area:** {m.median_crown_area_m2:.2f} m²",
        f"- **Median Detection Confidence:** {m.median_detection_confidence:.3f}",
        f"- **Low Confidence Detections (< 0.40):** {m.low_confidence_count}",
        "",
        "## Methodological Assumptions",
    ]
    for a in m.assumptions:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("## Quality Flags & Environmental Limitations")
    for qf in m.quality_flags:
        lines.append(f"- [Quality Flag] {qf}")
    for lim in m.limitations:
        lines.append(f"- [Limitation] {lim}")
    lines.append("")
    lines.append("## Methodological & Carbon Disclaimer")
    lines.append(m.carbon_disclaimer)
    return chr(10).join(lines)


def main():
    # -------------------------------------------------------------------------
    # 1. PRODUCT HERO & WORKFLOW STEPPER
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">FOREST CROWN AI</div>
            <div class="hero-subtitle">Individual tree crown detection and canopy measurement from high-resolution imagery.</div>
            <div class="hero-tagline">"Turn high-resolution forest imagery into measurable crown-level intelligence."</div>
        </div>
        <div class="workflow-bar">
            <span class="workflow-step">01 DATA</span>
            <span class="workflow-arrow">→</span>
            <span class="workflow-step">02 VALIDATE</span>
            <span class="workflow-arrow">→</span>
            <span class="workflow-step">03 ANALYZE</span>
            <span class="workflow-arrow">→</span>
            <span class="workflow-step">04 INSPECT</span>
            <span class="workflow-arrow">→</span>
            <span class="workflow-step">05 EXPORT</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # 2. ANALYSIS INPUT CARD (MAIN PAGE — FULL WIDTH)
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="section-title">ANALYSIS INPUT</div>
        <div class="section-sub">Select demonstration benchmark data or supply custom forest imagery and vector boundary.</div>
        """,
        unsafe_allow_html=True
    )

    image_bytes: Optional[bytes] = None
    aoi_bytes: Optional[bytes] = None
    image_name: str = ""
    user_gsd: Optional[float] = None
    geotiff_meta = None

    # Top Row: Dataset selector dropdown
    data_source_mode = st.selectbox(
        "DATASET",
        options=["Built-in Demo Dataset", "Upload Custom Files"],
        index=0,
        help="Select demonstration dataset or upload your own imagery and boundary."
    )

    if data_source_mode == "Built-in Demo Dataset":
        col_demo_sel, col_demo_badge = st.columns([1, 2])
        with col_demo_sel:
            demo_choice = st.selectbox(
                "DEMO DATASET",
                options=[
                    "Ordway-Swisher Forest Demo",
                    "Aerial Photo Demo (PNG)"
                ],
                index=0,
                help="Select pre-loaded demonstration forest imagery."
            )
        with col_demo_badge:
            if demo_choice == "Ordway-Swisher Forest Demo":
                tif_path = Path("demo_data/sample_forest_utm17n.tif")
                kml_path = Path("demo_data/sample_boundary.kml")
                if tif_path.exists():
                    image_bytes = tif_path.read_bytes()
                    image_name = tif_path.name
                if kml_path.exists():
                    aoi_bytes = kml_path.read_bytes()
                st.markdown(
                    """
                    <div class="benchmark-badge">
                        <span class="benchmark-badge-tag">Ordway-Swisher Forest Demo:</span>
                        Georeferenced NEON Forest (EPSG:32617, GSD: 0.10 m/px)
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                png_path = Path("demo_data/sample_aerial_photo.png")
                if png_path.exists():
                    image_bytes = png_path.read_bytes()
                    image_name = png_path.name
                user_gsd = 0.10
                st.markdown(
                    """
                    <div class="benchmark-badge">
                        <span class="benchmark-badge-tag">Aerial Photo Demo:</span>
                        Calibrated GSD 0.10 m/px (Local Metric Space)
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        # Two clean upload cards side by side
        col_card_a, col_card_b = st.columns(2)
        with col_card_a:
            st.markdown('<div class="upload-card-title">CARD A: FOREST BOUNDARY (.kml, .kmz)</div>', unsafe_allow_html=True)
            uploaded_aoi = st.file_uploader(
                "Upload Forest Boundary (KML/KMZ)",
                type=["kml", "kmz"],
                key="aoi_upload",
                help="Optional boundary defining Area of Interest (AOI)."
            )
            if uploaded_aoi:
                aoi_bytes = uploaded_aoi.getvalue()

        with col_card_b:
            st.markdown('<div class="upload-card-title">CARD B: FOREST IMAGERY (.tif, .tiff, .jpg, .jpeg, .png)</div>', unsafe_allow_html=True)
            uploaded_image = st.file_uploader(
                "Upload Forest Imagery",
                type=["tif", "tiff", "png", "jpg", "jpeg"],
                key="img_upload",
                help="GeoTIFF recommended for embedded CRS and GSD."
            )
            if uploaded_image:
                image_bytes = uploaded_image.getvalue()
                image_name = uploaded_image.name

        # Spatial Resolution / GSD Handling
        if image_bytes and image_name.lower().endswith((".tif", ".tiff")):
            try:
                geotiff_meta = read_geotiff_metadata(image_bytes)
                if geotiff_meta.has_georeference:
                    st.markdown(
                        f"""
                        <div class="benchmark-badge" style="margin-top: 4px;">
                            <span class="benchmark-badge-tag">Detected GSD:</span>
                            {geotiff_meta.gsd_x_m:.4f} × {geotiff_meta.gsd_y_m:.4f} m/px &nbsp;|&nbsp; 
                            <strong>Source:</strong> GeoTIFF metadata &nbsp;|&nbsp; 
                            <strong>CRS:</strong> {geotiff_meta.crs}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    user_gsd = None
                else:
                    st.warning("GeoTIFF has no embedded geotransform. Manual GSD is required.")
                    user_gsd = st.number_input(
                        "GROUND SAMPLING DISTANCE (M/PX)",
                        min_value=0.001,
                        max_value=10.0,
                        value=0.10,
                        step=0.01,
                        format="%.3f",
                        help="User-supplied spatial resolution."
                    )
                    st.caption("Provenance: User-supplied spatial resolution")
            except Exception:
                user_gsd = st.number_input(
                    "GROUND SAMPLING DISTANCE (M/PX)",
                    min_value=0.001,
                    max_value=10.0,
                    value=0.10,
                    step=0.01,
                    format="%.3f",
                    help="User-supplied spatial resolution."
                )
                st.caption("Provenance: User-supplied spatial resolution")
        elif image_bytes:
            user_gsd = st.number_input(
                "GROUND SAMPLING DISTANCE (M/PX)",
                min_value=0.001,
                max_value=10.0,
                value=float(user_gsd or 0.10),
                step=0.01,
                format="%.3f",
                help="User-supplied spatial resolution."
            )
            st.caption("Provenance: User-supplied spatial resolution")

    # If GeoTIFF metadata not read yet, read it
    if image_bytes and image_name.lower().endswith((".tif", ".tiff")) and geotiff_meta is None:
        try:
            geotiff_meta = read_geotiff_metadata(image_bytes)
        except Exception:
            geotiff_meta = None

    # Control parameters row
    col_conf, col_tile = st.columns(2)
    with col_conf:
        confidence_threshold = st.slider(
            "CONFIDENCE THRESHOLD",
            min_value=0.05,
            max_value=0.90,
            value=0.20,
            step=0.05,
            help="Minimum detector score threshold."
        )
    with col_tile:
        patch_size = st.slider(
            "TILING WINDOW (PX)",
            min_value=200,
            max_value=1200,
            value=400,
            step=50,
            help="Window size for sliding-window inference across the image."
        )

    # Prominent Centered Primary Button
    col_btn_l, col_btn_c, col_btn_r = st.columns([1.2, 1.6, 1.2])
    with col_btn_c:
        run_analysis_clicked = st.button("ANALYZE FOREST", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # 3. INPUT VALIDATION STATUS PANEL
    # -------------------------------------------------------------------------
    val_image = bool(image_bytes)
    val_boundary = bool(aoi_bytes)
    val_crs = False
    val_gsd = False
    val_overlap = True
    fix_reasons: List[str] = []

    # CRS & GSD evaluation
    if geotiff_meta and geotiff_meta.has_georeference:
        val_crs = True
        crs_label = f"✓ {geotiff_meta.crs}"
        val_gsd = True
        gsd_label = f"✓ {geotiff_meta.gsd_x_m:.2f} m/px"
    elif user_gsd is not None and user_gsd > 0:
        val_crs = False
        crs_label = "✓ Local Metric"
        val_gsd = True
        gsd_label = f"✓ {user_gsd:.2f} m/px"
    else:
        val_crs = False
        crs_label = "None"
        val_gsd = False
        gsd_label = "Missing"
        if val_image:
            fix_reasons.append("Provide a valid GSD (meters/pixel) for standard imagery.")

    if not val_image:
        fix_reasons.append("Select a demo dataset or upload forest imagery.")

    # AOI Overlap evaluation
    if val_image and val_boundary and geotiff_meta and geotiff_meta.has_georeference:
        try:
            aoi_poly, _ = load_kml_or_kmz(aoi_bytes)
            overlap_res = check_aoi_image_overlap(
                image_bounds=geotiff_meta.bounds,
                image_crs=geotiff_meta.crs,
                aoi_geom=aoi_poly,
                aoi_crs="EPSG:4326"
            )
            if overlap_res.status == OverlapStatus.NO_OVERLAP:
                val_overlap = False
                overlap_label = "❌ No Overlap"
                fix_reasons.append("Boundary lies completely outside image extent. Ensure both datasets match location.")
            else:
                val_overlap = True
                overlap_label = "✓ Verified"
        except Exception as e:
            val_overlap = False
            overlap_label = "❌ Error"
            fix_reasons.append(f"Failed to parse boundary: {e}")
    elif val_boundary:
        val_overlap = True
        overlap_label = "✓ Verified"
    else:
        val_overlap = True
        overlap_label = "✓ Full Extent"

    ready_for_analysis = val_image and val_gsd and val_overlap

    # Render Compact Validation Card
    status_badge = (
        '<span class="val-badge-ready">READY FOR ANALYSIS</span>'
        if ready_for_analysis else
        '<span class="val-badge-blocked">ACTION REQUIRED</span>'
    )

    st.markdown(
        f"""
        <div class="validation-card">
            <div class="val-header">INPUT VALIDATION</div>
            <div class="val-grid">
                <div class="val-col">
                    <span class="val-lbl">IMAGE</span>
                    <span class="val-txt">{'✓ Loaded' if val_image else 'Missing'}</span>
                </div>
                <div class="val-col">
                    <span class="val-lbl">BOUNDARY</span>
                    <span class="val-txt">{'✓ Loaded' if val_boundary else '✓ Full Extent'}</span>
                </div>
                <div class="val-col">
                    <span class="val-lbl">CRS</span>
                    <span class="val-txt">{crs_label}</span>
                </div>
                <div class="val-col">
                    <span class="val-lbl">GSD</span>
                    <span class="val-txt">{gsd_label}</span>
                </div>
                <div class="val-col">
                    <span class="val-lbl">AOI OVERLAP</span>
                    <span class="val-txt">{overlap_label}</span>
                </div>
                <div class="val-col">
                    <span class="val-lbl">STATUS</span>
                    {status_badge}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not ready_for_analysis:
        for fr in fix_reasons:
            st.error(f"Validation Issue: {fr}")
        if run_analysis_clicked:
            st.stop()

    # -------------------------------------------------------------------------
    # 4. ANALYSIS EXECUTION
    # -------------------------------------------------------------------------
    if run_analysis_clicked:
        if not ready_for_analysis:
            st.error("Cannot proceed: resolve validation issues above.")
            return

        with st.spinner("Executing DeepForest neural detection, geometric clipping, and spatial dissolution..."):
            try:
                detector = get_cached_deepforest_detector()
                result: PipelineResult = run_forest_crown_pipeline(
                    image_source=image_bytes,
                    user_gsd=user_gsd,
                    aoi_source=aoi_bytes,
                    confidence_threshold=confidence_threshold,
                    patch_size=patch_size,
                    patch_overlap=0.15,
                    export_dir="./outputs",
                    export_base_name="forest_crown_analysis",
                    detector=detector,
                    use_mock_detector=False
                )
                st.session_state["result"] = result
                st.session_state["image_name"] = image_name

            except MissingGSDError as e:
                st.error(f"Spatial Resolution Error: {str(e)}")
                return
            except GeospatialError as e:
                st.error(f"Geospatial Error: {str(e)}")
                return
            except Exception as e:
                st.error(f"Analysis Processing Error: {str(e)}")
                return

    # -------------------------------------------------------------------------
    # 5. RESULTS DASHBOARD (FULL WIDTH)
    # -------------------------------------------------------------------------
    if "result" in st.session_state:
        result: PipelineResult = st.session_state["result"]
        img_name = st.session_state.get("image_name", "Imagery")
        m = result.metrics

        st.markdown(
            """
            <div class="section-title">FOREST ANALYSIS COMPLETE</div>
            <div class="section-sub">DeepForest crown detection and canopy assessment</div>
            """,
            unsafe_allow_html=True
        )

        # Six Professional KPI Cards
        forest_area_str = f"{m.forest_area_m2:,.1f} m²" if m.forest_area_m2 else "Full Raster"
        forest_ha_str = f"{m.forest_area_ha:.4f} ha" if m.forest_area_ha else "Entire Extent"
        unique_canopy_str = f"{m.unique_canopy_area_m2:,.1f} m²"
        unique_ha_str = f"{m.dissolved_canopy_area_ha:.4f} ha"
        cover_str = f"{m.canopy_cover_pct:.1f}%" if m.canopy_cover_pct is not None else "N/A"

        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-label">TREES DETECTED</div>
                    <div class="kpi-value">{m.tree_count:,}</div>
                    <div class="kpi-sub">Distinct Crowns</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">FOREST AREA</div>
                    <div class="kpi-value">{forest_area_str}</div>
                    <div class="kpi-sub">{forest_ha_str}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">UNIQUE CANOPY AREA</div>
                    <div class="kpi-value">{unique_canopy_str}</div>
                    <div class="kpi-sub">{unique_ha_str}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">CANOPY COVER</div>
                    <div class="kpi-value">{cover_str}</div>
                    <div class="kpi-sub">Ground Footprint</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">MEAN CROWN AREA</div>
                    <div class="kpi-value">{m.mean_crown_area_m2:.2f} m²</div>
                    <div class="kpi-sub">Average Dimension</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">MEDIAN CROWN AREA</div>
                    <div class="kpi-value">{m.median_crown_area_m2:.2f} m²</div>
                    <div class="kpi-sub">Median Dimension</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------------------------------------------------------------------
        # 6. CROWN DETECTION MAP CENTERPIECE (650PX HEIGHT)
        # ---------------------------------------------------------------------
        st.markdown(
            """
            <div class="section-title">CROWN DETECTION MAP</div>
            <div class="section-sub">Detected crown geometries overlaid on source imagery.</div>
            """,
            unsafe_allow_html=True
        )

        if result.metadata.has_georeference and result.metadata.bounds:
            if result.aoi_wgs84_geom and not result.aoi_wgs84_geom.is_empty:
                center_lat = float(result.aoi_wgs84_geom.centroid.y)
                center_lon = float(result.aoi_wgs84_geom.centroid.x)
            else:
                from pyproj import Transformer
                cx = (result.metadata.bounds[0] + result.metadata.bounds[2]) / 2.0
                cy = (result.metadata.bounds[1] + result.metadata.bounds[3]) / 2.0
                transformer = Transformer.from_crs(result.metadata.crs or "EPSG:32617", "EPSG:4326", always_xy=True)
                center_lon, center_lat = transformer.transform(cx, cy)

            crown_areas = [d.approx_area_m2(m.gsd_m) for d in result.detections]
            tree_ids = [d.detection_id for d in result.detections]

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
            if folium_map:
                from streamlit_folium import st_folium
                st_folium(folium_map, width=1380, height=650)
        else:
            st.markdown("##### High-Resolution Local Detections")
            annotated_img = overlay_detections_on_image(
                image_rgb=result.image_rgb,
                detections=result.detections,
                line_width=2
            )
            st.image(
                annotated_img,
                caption=f"DeepForest 2.1.0 detections ({len(result.detections)} crowns mapped in local metric space).",
                use_container_width=True
            )

        # ---------------------------------------------------------------------
        # 7. ANALYTICS SECTION (PLOTLY CHARTS SIDE BY SIDE)
        # ---------------------------------------------------------------------
        st.markdown(
            """
            <div class="section-title">ANALYTICS</div>
            <div class="section-sub">Crown size and detection confidence distributions</div>
            """,
            unsafe_allow_html=True
        )

        crown_areas_all = [d.approx_area_m2(m.gsd_m) for d in result.detections]
        confidences_all = [d.confidence for d in result.detections]

        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            fig_size = plot_crown_size_distribution_dark(crown_areas_all)
            if fig_size:
                st.plotly_chart(fig_size, use_container_width=True)
        with col_ch2:
            fig_conf = plot_detection_confidence_dark(confidences_all)
            if fig_conf:
                st.plotly_chart(fig_conf, use_container_width=True)

        # ---------------------------------------------------------------------
        # 8. DETECTION INSPECTION TABLE
        # ---------------------------------------------------------------------
        st.markdown(
            """
            <div class="section-title">DETECTION INSPECTION</div>
            <div class="section-sub">Tabular record of individual tree crown detections.</div>
            """,
            unsafe_allow_html=True
        )

        if result.detections:
            rows = []
            for d in result.detections:
                area_val = d.approx_area_m2(m.gsd_m)
                # Approximate equivalent circular diameter: d = 2 * sqrt(Area / pi)
                approx_diam_m = 2.0 * math.sqrt(max(0.0, area_val) / math.pi) if area_val > 0 else 0.0

                rows.append({
                    "Tree ID": d.detection_id,
                    "Crown Area (m²)": f"{area_val:.2f}",
                    "Confidence Score": f"{d.confidence:.3f}",
                    "Approx Diameter (m)": f"{approx_diam_m:.2f}"
                })
            df_crowns = pd.DataFrame(rows)
            st.dataframe(df_crowns, use_container_width=True, height=280)
        else:
            st.info("No crowns detected above threshold.")

        # ---------------------------------------------------------------------
        # 9. QUALITY & INTERPRETATION AND MODEL LIMITATIONS (SIDE BY SIDE)
        # ---------------------------------------------------------------------
        col_quality, col_limits = st.columns(2)

        with col_quality:
            st.markdown(
                """
                <div class="section-title" style="font-size: 0.96rem;">QUALITY & INTERPRETATION</div>
                <div class="section-sub">Sensor provenance, projection and geometric integrity.</div>
                """,
                unsafe_allow_html=True
            )
            overlap_pct_str = f"{m.overlap_redundancy_pct:.1f}% ({m.overlap_redundancy_m2:,.1f} m²)"
            st.markdown(
                f"""
                <div class="quality-card">
                    <div class="quality-row">
                        <span class="quality-key">GSD Provenance</span>
                        <span class="quality-val">✓ Verified ({m.gsd_source})</span>
                    </div>
                    <div class="quality-row">
                        <span class="quality-key">Coordinate Reference</span>
                        <span class="quality-val">✓ Metric ({m.crs})</span>
                    </div>
                    <div class="quality-row">
                        <span class="quality-key">AOI Alignment</span>
                        <span class="quality-val">✓ Verified</span>
                    </div>
                    <div class="quality-row">
                        <span class="quality-key">Detection Threshold</span>
                        <span class="quality-val">{m.confidence_threshold:.2f}</span>
                    </div>
                    <div class="quality-row">
                        <span class="quality-key">Low-Confidence Count</span>
                        <span class="quality-val">{m.low_confidence_count} crowns (< 0.40)</span>
                    </div>
                    <div class="quality-row">
                        <span class="quality-key">Overlap Redundancy</span>
                        <span class="quality-val">{overlap_pct_str}</span>
                    </div>
                    <div class="quality-row">
                        <span class="quality-key">Crown Geometry</span>
                        <span class="quality-val">Inscribed Ellipse Approx.</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_limits:
            st.markdown(
                """
                <div class="section-title" style="font-size: 0.96rem;">MODEL LIMITATIONS</div>
                <div class="section-sub">Environmental, optical and allometric constraints.</div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <div class="limitations-card">
                    <div class="limitations-item"><span class="limitations-bullet">•</span><strong>Model Predictions:</strong> Detections are neural model inferences, not direct ground-truth field surveys.</div>
                    <div class="limitations-item"><span class="limitations-bullet">•</span><strong>Interlocking Canopy:</strong> Dense or touching crowns may be merged into single detection envelopes.</div>
                    <div class="limitations-item"><span class="limitations-bullet">•</span><strong>Canopy Shadows:</strong> Deep cast shadows can create false negatives in lower understory layers.</div>
                    <div class="limitations-item"><span class="limitations-bullet">•</span><strong>Understory Saplings:</strong> Smaller juvenile trees are occluded by dominant overstory crowns.</div>
                    <div class="limitations-item"><span class="limitations-bullet">•</span><strong>Geometric Approximation:</strong> Crown boundaries are modeled as inscribed ellipses, not full instance segmentations.</div>
                    <div class="limitations-item"><span class="limitations-bullet">•</span><strong>Not Biomass / Not Carbon:</strong> Optical horizontal canopy area does not measure wood density, tree height, or trunk volume. Field allometry and independent auditing are required for carbon accounting.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---------------------------------------------------------------------
        # 10. METHODOLOGY PIPELINE
        # ---------------------------------------------------------------------
        st.markdown(
            """
            <div class="section-title">METHODOLOGY</div>
            <div class="section-sub">Eight-stage deterministic canopy quantification pipeline.</div>
            <div class="method-grid">
                <div class="method-step">
                    <div class="method-num">Stage 01</div>
                    <strong>Load & Validate Imagery</strong><br>
                    Verify band depth, spatial extent, and affine geotransform resolution.
                </div>
                <div class="method-step">
                    <div class="method-num">Stage 02</div>
                    <strong>Align Forest Boundary</strong><br>
                    Parse KML/KMZ and reproject coordinate frames to match raster CRS.
                </div>
                <div class="method-step">
                    <div class="method-num">Stage 03</div>
                    <strong>Clip Imagery to AOI</strong><br>
                    Crop raster, update bounding transform, and mask out-of-boundary pixels.
                </div>
                <div class="method-step">
                    <div class="method-num">Stage 04</div>
                    <strong>DeepForest Detection</strong><br>
                    Execute windowed neural inference with boundary Non-Maximum Suppression.
                </div>
                <div class="method-step">
                    <div class="method-num">Stage 05</div>
                    <strong>Construct Crown Geometries</strong><br>
                    Model crowns as inscribed ellipses to eliminate rectangular bounding error.
                </div>
                <div class="method-step">
                    <div class="method-num">Stage 06</div>
                    <strong>Dissolve Overlapping Canopy</strong><br>
                    Execute spatial unary union to compute true non-overlapping canopy footprint.
                </div>
                <div class="method-step">
                    <div class="method-num">Stage 07</div>
                    <strong>Calculate Canopy Metrics</strong><br>
                    Quantify true coverage ratio, stand density, and overlap redundancy.
                </div>
                <div class="method-step">
                    <div class="method-num">Stage 08</div>
                    <strong>Generate GIS Outputs</strong><br>
                    Export interoperable GeoJSON, Shapefile, CSV, and summary documentation.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------------------------------------------------------------------
        # 11. EXPORT ANALYSIS (CENTERED DOWNLOAD BUTTONS)
        # -------------------------------------------------------------------------
        st.markdown(
            """
            <div class="section-title">EXPORT ANALYSIS</div>
            <div class="section-sub">Download crown geometries, measurements and analysis metadata.</div>
            """,
            unsafe_allow_html=True
        )

        exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)

        csv_path = result.exported_files.get("crowns_csv")
        if csv_path and Path(csv_path).exists():
            with open(csv_path, "rb") as f:
                exp_c1.download_button(
                    label="CSV",
                    data=f.read(),
                    file_name="forest_crown_detections.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        geojson_path = result.exported_files.get("crowns_geojson")
        if geojson_path and Path(geojson_path).exists():
            with open(geojson_path, "rb") as f:
                exp_c2.download_button(
                    label="GeoJSON",
                    data=f.read(),
                    file_name="forest_crown_geometries.geojson",
                    mime="application/geo+json",
                    use_container_width=True
                )

        kml_path = result.exported_files.get("crowns_kml")
        if kml_path and Path(kml_path).exists():
            with open(kml_path, "rb") as f:
                exp_c3.download_button(
                    label="KML",
                    data=f.read(),
                    file_name="forest_crowns_google_earth.kml",
                    mime="application/vnd.google-earth.kml+xml",
                    use_container_width=True
                )

        summary_text = generate_summary_report(result, img_name)
        exp_c4.download_button(
            label="SUMMARY REPORT",
            data=summary_text,
            file_name="canopy_analysis_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    else:
        # Compact Ready for Analysis Callout (Before Analysis)
        st.markdown(
            """
            <div class="ready-callout">
                <div class="callout-badge">READY FOR ANALYSIS</div>
                <div class="callout-content">
                    <div class="callout-title">Demo imagery and forest boundary validated successfully.</div>
                    <div class="callout-sub">Click <strong>ANALYZE FOREST</strong> above to execute DeepForest neural detection, inscribed crown modeling, and continuous canopy footprint dissolution.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------------------
    # 12. PROFESSIONAL FOOTER
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="app-footer">
            <span><strong>Forest Crown AI</strong> — High-resolution forest analytics</span>
            <span>Model predictions should be validated before operational use.</span>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
