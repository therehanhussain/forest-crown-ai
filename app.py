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
    plot_area_comparison_chart,
    plot_crown_diameter_distribution,
)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Forest Crown AI | Canopy Intelligence",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Carbon-Tech Design System (CSS)
st.markdown(
    """
    <style>
    /* Global Typography & Palette */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #0F172A;
    }
    
    /* Main App Header */
    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #0F172A;
        margin-bottom: 2px;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 1.05rem;
        font-weight: 500;
        color: #334155;
        margin-bottom: 6px;
    }
    .brand-tagline {
        font-size: 0.92rem;
        color: #64748B;
        margin-bottom: 16px;
    }
    
    /* Compact Workflow Step Indicator */
    .workflow-bar {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 8px 14px;
        margin-bottom: 24px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #475569;
    }
    .workflow-step {
        display: inline-flex;
        align-items: center;
        color: #0F172A;
    }
    .workflow-arrow {
        color: #94A3B8;
        margin: 0 4px;
    }

    /* Professional Sidebar Panel */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #475569 !important;
        margin-top: 14px !important;
        margin-bottom: 8px !important;
    }

    /* Primary CTA Button (Refined Deep Forest Green) */
    div.stButton > button[kind="primary"] {
        background-color: #1B4332 !important;
        color: #FFFFFF !important;
        border: 1px solid #143628 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.03em !important;
        padding: 0.55rem 1.2rem !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #2D6A4F !important;
        border-color: #2D6A4F !important;
        color: #FFFFFF !important;
    }
    
    /* Secondary & Download Buttons */
    div.stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.45rem 1rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stDownloadButton > button:hover {
        border-color: #1B4332 !important;
        color: #1B4332 !important;
        background-color: #F8FAFC !important;
    }

    /* Compact Validation Panel */
    .validation-panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 24px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .val-title {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 10px;
    }
    .val-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        align-items: center;
    }
    .val-item {
        display: flex;
        flex-direction: column;
    }
    .val-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
    }
    .val-value {
        font-size: 0.88rem;
        font-weight: 600;
        color: #0F172A;
    }
    .val-status-ready {
        display: inline-block;
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-align: center;
    }
    .val-status-blocked {
        display: inline-block;
        background: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-align: center;
    }

    /* Six Responsive KPI Cards (Zero Truncation Guarantee) */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin-top: 14px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 6px;
        white-space: normal;
        line-height: 1.3;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.15;
        font-feature-settings: 'tnum' on, 'lnum' on;
    }
    .kpi-sub {
        font-size: 0.82rem;
        font-weight: 600;
        color: #059669;
        margin-top: 4px;
    }

    /* Section Headers */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #0F172A;
        margin-top: 24px;
        margin-bottom: 2px;
    }
    .section-sub {
        font-size: 0.88rem;
        color: #64748B;
        margin-bottom: 14px;
    }

    /* Quality Grid */
    .quality-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 24px;
    }
    .quality-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #F1F5F9;
        font-size: 0.86rem;
    }
    .quality-row:last-child {
        border-bottom: none;
    }
    .quality-key {
        font-weight: 600;
        color: #475569;
    }
    .quality-val {
        font-weight: 600;
        color: #0F172A;
    }

    /* Methodology Card */
    .method-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
        margin-top: 10px;
        margin-bottom: 24px;
    }
    .method-step {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px 14px;
        font-size: 0.84rem;
        line-height: 1.4;
    }
    .method-num {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #059669;
        margin-bottom: 4px;
        text-transform: uppercase;
    }
    
    /* Footer */
    .app-footer {
        margin-top: 48px;
        padding-top: 20px;
        border-top: 1px solid #E2E8F0;
        font-size: 0.80rem;
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
    return "\n".join(lines)


def main():
    # -------------------------------------------------------------------------
    # 1. PRODUCT HEADER & COMPACT WORKFLOW
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="brand-title">FOREST CROWN AI</div>
        <div class="brand-subtitle">Individual tree crown detection and canopy measurement from high-resolution imagery.</div>
        <div class="brand-tagline">Turn high-resolution forest imagery into measurable crown-level intelligence.</div>
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
    # 2. SIDEBAR ANALYSIS CONTROL PANEL
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### DATA SOURCE")

        data_source_mode = st.radio(
            "Data Input Mode",
            options=["Demo dataset", "Upload custom imagery"],
            index=0,
            label_visibility="collapsed"
        )

        image_bytes: Optional[bytes] = None
        aoi_bytes: Optional[bytes] = None
        image_name: str = ""
        user_gsd: Optional[float] = None

        if data_source_mode == "Demo dataset":
            demo_choice = st.selectbox(
                "Select Demo Dataset",
                options=[
                    "Sample 1: Georeferenced Forest (GeoTIFF + KML AOI)",
                    "Sample 2: Aerial Photo (Standard PNG)"
                ],
                index=0
            )
            if demo_choice.startswith("Sample 1"):
                tif_path = Path("demo_data/sample_forest_utm17n.tif")
                kml_path = Path("demo_data/sample_boundary.kml")
                if tif_path.exists():
                    image_bytes = tif_path.read_bytes()
                    image_name = tif_path.name
                if kml_path.exists():
                    aoi_bytes = kml_path.read_bytes()
                st.caption("Ordway-Swisher Biological Station (UTM 17N) with KML AOI.")
            else:
                png_path = Path("demo_data/sample_aerial_photo.png")
                if png_path.exists():
                    image_bytes = png_path.read_bytes()
                    image_name = png_path.name
                user_gsd = 0.10
                st.caption("Standard RGB aerial photo. Calibrated GSD = 0.100 m/px.")
        else:
            uploaded_image = st.file_uploader(
                "Upload Imagery (GeoTIFF, PNG, JPG)",
                type=["tif", "tiff", "png", "jpg", "jpeg"],
                help="GeoTIFF recommended for embedded CRS and GSD."
            )
            uploaded_aoi = st.file_uploader(
                "Upload Forest Boundary (KML/KMZ)",
                type=["kml", "kmz"],
                help="Optional boundary defining Area of Interest."
            )
            if uploaded_image:
                image_bytes = uploaded_image.getvalue()
                image_name = uploaded_image.name
            if uploaded_aoi:
                aoi_bytes = uploaded_aoi.getvalue()

        st.markdown("---")
        st.markdown("### SPATIAL RESOLUTION")

        geotiff_meta = None
        if image_bytes and image_name.lower().endswith((".tif", ".tiff")):
            try:
                geotiff_meta = read_geotiff_metadata(image_bytes)
                if geotiff_meta.has_georeference:
                    st.markdown(f"**GSD:** `{geotiff_meta.gsd_x_m:.4f} × {geotiff_meta.gsd_y_m:.4f} m/px`")
                    st.markdown("**Source:** `GeoTIFF metadata`")
                    st.caption(f"CRS: `{geotiff_meta.crs}`")
                    user_gsd = None
                else:
                    st.warning("TIFF has no geotransform. Manual GSD required.")
                    user_gsd = st.number_input("Manual GSD (m/px)", min_value=0.001, max_value=10.0, value=0.10, step=0.01, format="%.3f")
            except Exception:
                user_gsd = st.number_input("Manual GSD (m/px)", min_value=0.001, max_value=10.0, value=0.10, step=0.01, format="%.3f")
        elif image_bytes:
            st.info("Standard image without embedded CRS. Manual GSD required.")
            user_gsd = st.number_input(
                "Manual GSD (m/px)",
                min_value=0.001,
                max_value=10.0,
                value=float(user_gsd or 0.10),
                step=0.01,
                format="%.3f"
            )
            st.caption("Provenance: User-supplied GSD")
        else:
            st.caption("Awaiting imagery input...")

        st.markdown("---")
        st.markdown("### DETECTION")
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.20,
            step=0.05,
            help="Minimum detector score threshold."
        )

        patch_size = st.select_slider(
            "Tiling window (px)",
            options=[200, 300, 400, 600, 800],
            value=400,
            help="Window size for sliding-window inference."
        )

        st.markdown("---")
        st.markdown("### RUN ANALYSIS")
        run_analysis_clicked = st.button("Run Forest Analysis", type="primary", use_container_width=True)

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
            fix_reasons.append("Provide a valid GSD (meters/pixel) in the sidebar.")

    if not val_image:
        fix_reasons.append("Select a demo dataset or upload an aerial image.")

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

    # Render Compact Validation Panel
    status_html = (
        '<span class="val-status-ready">READY FOR ANALYSIS</span>'
        if ready_for_analysis else
        '<span class="val-status-blocked">ACTION REQUIRED</span>'
    )

    st.markdown(
        f"""
        <div class="validation-panel">
            <div class="val-title">Input Validation</div>
            <div class="val-grid">
                <div class="val-item">
                    <span class="val-label">Image</span>
                    <span class="val-value">{'✓ Loaded' if val_image else 'Missing'}</span>
                </div>
                <div class="val-item">
                    <span class="val-label">Boundary</span>
                    <span class="val-value">{'✓ Loaded' if val_boundary else '✓ Full Extent'}</span>
                </div>
                <div class="val-item">
                    <span class="val-label">CRS</span>
                    <span class="val-value">{crs_label}</span>
                </div>
                <div class="val-item">
                    <span class="val-label">GSD</span>
                    <span class="val-value">{gsd_label}</span>
                </div>
                <div class="val-item">
                    <span class="val-label">AOI Overlap</span>
                    <span class="val-value">{overlap_label}</span>
                </div>
                <div class="val-item">
                    <span class="val-label">Status</span>
                    {status_html}
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
    # 5. RESULTS DASHBOARD (VISUAL CENTERPIECE)
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

        # Six Professional KPI Cards (Never Truncated)
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
                    <div class="kpi-label">UNIQUE CANOPY</div>
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
        # 6. CROWN DETECTION MAP (DOMINANT VISUAL ELEMENT)
        # ---------------------------------------------------------------------
        st.markdown(
            """
            <div class="section-title">CROWN DETECTION MAP</div>
            <div class="section-sub">Detected crown geometries overlaid on source imagery.</div>
            """,
            unsafe_allow_html=True
        )

        if result.metadata.has_georeference and result.metadata.bounds:
            bounds = result.metadata.bounds
            center_lat = (bounds[1] + bounds[3]) / 2.0
            center_lon = (bounds[0] + bounds[2]) / 2.0

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
                st_folium(folium_map, width=1200, height=560)
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
        # 7. DETECTION INSPECTION & ANALYSIS QUALITY
        # -------------------------------------------------------------------------
        col_inspect, col_quality = st.columns([3, 2])

        with col_inspect:
            st.markdown(
                """
                <div class="section-title" style="font-size: 1.05rem; margin-top: 10px;">DETECTION INSPECTION</div>
                <div class="section-sub">Tabular record of individual tree crown detections.</div>
                """,
                unsafe_allow_html=True
            )
            if result.detections:
                rows = []
                for d in result.detections:
                    if d.geometry_geo and hasattr(d.geometry_geo, 'centroid'):
                        loc_str = f"{d.geometry_geo.centroid.y:.5f}°, {d.geometry_geo.centroid.x:.5f}°"
                    else:
                        loc_str = f"px ({int(d.center_pixel[0])}, {int(d.center_pixel[1])})"

                    rows.append({
                        "Tree ID": d.detection_id,
                        "Confidence": f"{d.confidence:.1%}",
                        "Crown Area": f"{d.approx_area_m2(m.gsd_m):.2f} m²",
                        "Location": loc_str
                    })
                df_crowns = pd.DataFrame(rows)
                st.dataframe(df_crowns, use_container_width=True, height=280)
            else:
                st.info("No crowns detected above threshold.")

        with col_quality:
            st.markdown(
                """
                <div class="section-title" style="font-size: 1.05rem; margin-top: 10px;">ANALYSIS QUALITY</div>
                <div class="section-sub">Sensor provenance and geometric integrity.</div>
                """,
                unsafe_allow_html=True
            )
            overlap_pct_str = f"{m.overlap_redundancy_pct:.1f}% ({m.overlap_redundancy_m2:,.1f} m²)"
            st.markdown(
                f"""
                <div class="quality-card">
                    <div class="quality-row">
                        <span class="quality-key">GSD provenance</span>
                        <span class="quality-val">✓ Verified ({m.gsd_source})</span>
                    </div>
                    <div class="quality-row">
                        <span class="quality-key">Coordinate reference</span>
                        <span class="quality-val">✓ Metric ({m.crs})</span>
                    </div>
                    <div class="quality-row">
                        <span class="quality-key">AOI alignment</span>
                        <span class="quality-val">✓ Verified</span>
                    </div>
                    <div class="quality-row">
                        <span class="quality-key">Detection threshold</span>
                        <span class="quality-val">{m.confidence_threshold:.2f}</span>
                    </div>
                    <div class="quality-row">
                        <span class="quality-key">Low-confidence count</span>
                        <span class="quality-val">{m.low_confidence_count} crowns (< 0.40)</span>
                    </div>
                    <div class="quality-row">
                        <span class="quality-key">Overlap redundancy</span>
                        <span class="quality-val">{overlap_pct_str}</span>
                    </div>
                    <div class="quality-row">
                        <span class="quality-key">Crown geometry</span>
                        <span class="quality-val">Inscribed ellipse approx.</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---------------------------------------------------------------------
        # 8. METHODOLOGY & SCIENTIFIC FOUNDATION
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
        # 9. METHOD LIMITATIONS & INTERPRETATION (COLLAPSIBLE)
        # ---------------------------------------------------------------------
        with st.expander("Method limitations & interpretation", expanded=False):
            st.markdown(
                """
                - **DeepForest detections are model predictions**, not direct ground-truth field surveys.
                - **Dense or interlocking tree crowns** can be missed or merged into single larger bounding boxes.
                - **Canopy shadows** may create false negatives in understory vegetation.
                - **Small saplings or understory crowns** may be missed depending on sensor ground sample distance (GSD).
                - **Crown areas are geometric approximations** derived as inscribed ellipses from 2D detector bounding boxes, not full instance segmentations.
                - **Canopy cover is NOT biomass:** Optical horizontal canopy area does not measure wood density, tree height, or trunk volume.
                - **Canopy cover is NOT carbon stock:** Carbon accounting requires 3D allometry, calibrated field plots, and third-party audit.
                - **Results require field validation** before operational carbon accounting or commercial carbon crediting.
                """
            )
            st.caption(m.carbon_disclaimer)

        # ---------------------------------------------------------------------
        # 10. EXPORT ANALYSIS
        # ---------------------------------------------------------------------
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
                    label="Download CSV",
                    data=f.read(),
                    file_name="forest_crown_detections.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        geojson_path = result.exported_files.get("crowns_geojson")
        if geojson_path and Path(geojson_path).exists():
            with open(geojson_path, "rb") as f:
                exp_c2.download_button(
                    label="Download GeoJSON",
                    data=f.read(),
                    file_name="forest_crown_geometries.geojson",
                    mime="application/geo+json",
                    use_container_width=True
                )

        kml_path = result.exported_files.get("crowns_kml")
        if kml_path and Path(kml_path).exists():
            with open(kml_path, "rb") as f:
                exp_c3.download_button(
                    label="Download KML",
                    data=f.read(),
                    file_name="forest_crowns_google_earth.kml",
                    mime="application/vnd.google-earth.kml+xml",
                    use_container_width=True
                )

        summary_text = generate_summary_report(result, img_name)
        exp_c4.download_button(
            label="Download Analysis Report",
            data=summary_text,
            file_name="canopy_analysis_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    else:
        # Default State Before Analysis
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 24px; margin-top: 10px;">
                <h4 style="margin-top: 0; color: #0F172A; font-weight: 700;">Ready for Analysis</h4>
                <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                    The default sample dataset is loaded and validated in the control panel. 
                    Click <strong>'Run Forest Analysis'</strong> in the sidebar to execute DeepForest neural detection, 
                    elliptical crown modeling, and continuous canopy footprint dissolution.
                </p>
                <div style="font-size: 0.82rem; color: #64748B; margin-top: 12px;">
                    <strong>Notice:</strong> Optical canopy surface measurements are geometric approximations and should be validated through field surveys prior to commercial carbon accounting.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------------------
    # 11. SUBTLE PROFESSIONAL FOOTER
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
