# 🌲 Forest Crown AI

> **Autonomous Individual Tree Crown (ITC) Detection & Canopy Area Estimation**  
> *Production-quality geospatial computer vision built on DeepForest, Rasterio, and Shapely.*

---

## 📖 Executive Summary & Engineering Philosophy

**Forest Crown AI** is a geospatial AI system designed to solve a core problem in environmental monitoring and forestry:

Given high-resolution aerial/drone imagery (GeoTIFF, PNG, JPG) and/or a KML forest boundary, the application:
1. **Detects and counts individual tree crowns** using RetinaNet deep neural networks (DeepForest).
2. **Estimates true canopy surface area** via spatial union dissolution.
3. **Presents results intuitively** so any forest manager or researcher can use the system without developer assistance.
4. **Transparently reports uncertainty, assumptions, and physical limitations**.
5. **Exports analysis-ready GIS datasets** (GeoJSON, Shapefile, CSV with WKT, Google Earth KML).

### ⚖️ Core Engineering Principles

| Principle | Implementation Rule |
| :--- | :--- |
| **No Fabricated Measurements** | The system **never silently invents GSD** (Ground Sample Distance). If an unreferenced JPG/PNG is uploaded, explicit user GSD entry is strictly mandatory. |
| **Preserve Spatial CRS** | Retains coordinate reference system metadata throughout the pipeline. Projections and affine transforms are never discarded. |
| **Metric Units Only** | Areas and distances are computed strictly in a **projected metric CRS** (e.g. UTM in meters/hectares), **never in latitude/longitude degrees**. |
| **No Naive Box Summation** | Overlapping tree crowns share canopy footprint. Simple bounding-box addition produces false double-counting. We dissolve crowns using **geometric unary union**. |
| **Separate Count from Area** | Crown count (discrete trees) is decoupled from continuous canopy cover ($m^2$ / hectares / percentage of AOI). |
| **Visible Assumptions** | Every report lists GSD source, projection code, confidence cutoffs, and edge effects. |
| **Zero Carbon Hype** | **We do not claim biomass, carbon stock, or verified carbon credits.** Optical 2D canopy surface area is physically incapable of verifying carbon credits without destructive sampling or calibrated LiDAR height profiles. |

---

## 🏗️ Architecture & Component Design

```
forest-crown-ai/
├── app.py                     # Streamlit self-service web interface & tabs
├── pipeline.py                # End-to-end processing orchestrator
├── requirements.txt           # Pinned production dependencies (Python 3.11)
├── README.md                  # System architecture & scientific documentation
├── .gitignore                 # Geospatial, raster, and virtualenv exclusions
├── src/
│   ├── __init__.py
│   ├── io_utils.py            # GeoTIFF metadata, GSD extraction, KML/KMZ parsing, GIS export
│   ├── preprocessing.py       # Array normalization, dynamic-range stretch, AOI clipping, tiling
│   ├── detection.py           # DeepForest RetinaNet integration, Non-Maximum Suppression (NMS)
│   ├── geometry.py            # UTM estimation, CRS reprojection, AOI overlap checking, spatial union
│   ├── metrics.py             # Canopy cover %, stand density, overlap redundancy, uncertainty
│   └── visualization.py       # PIL overlays, interactive Folium satellite maps, Plotly distributions
├── tests/                     # Comprehensive unit & integration test suite
│   ├── __init__.py
│   ├── test_io_utils.py
│   ├── test_geometry.py
│   ├── test_metrics.py
│   └── test_pipeline.py
├── data/                      # Sample raster & KML inputs (.gitkeep)
└── outputs/                   # Exported GeoJSON, Shapefiles, CSVs, KMLs (.gitkeep)
```

### Module Breakdown

#### 1. `src/io_utils.py`
- **Metadata Extraction**: Reads raster affine transform matrix `(a, b, c, d, e, f)` and calculates true Ground Sample Distance ($GSD_x = |a|$, $GSD_y = |e|$).
- **Geographic GSD Approximation**: When a GeoTIFF is in EPSG:4326 (degrees), calculates metric resolution at center latitude using WGS84 ellipsoidal distance equations.
- **Pure-Python KML/KMZ Loader**: Parses `<coordinates>` in `<outerBoundaryIs>` across any KML namespace without requiring external GDAL drivers. Unzips `.kmz` packages on the fly.
- **Geospatial Exporter**: Generates GeoJSON, Shapefile, CSV (with WKT centroids and bounding boxes), and Google Earth KML files.

#### 2. `src/preprocessing.py`
- **Dynamic Range Normalization**: Converts 16-bit satellite rasters, multispectral images, or RGBA photos into standard 3-channel RGB uint8 `[0, 255]` using a 2%–98% percentile contrast stretch.
- **AOI Clipping**: Uses inverse affine transforms to mask raster pixels outside the user-specified boundary.
- **Sliding-Window Tiling**: Slices large drone orthomosaics into configurable patches (default $400 \times 400$ px) with seam overlap.

#### 3. `src/detection.py`
- **DeepForest Model**: Loads Weinstein et al.'s pre-trained tree crown detection model (RetinaNet with ResNet50 backbone).
- **Non-Maximum Suppression (NMS)**: Eliminates duplicate detections generated across overlapping tile seams.
- **Mock Detector Fallback**: Enables rapid testing in headless CI or offline environments without multi-gigabyte neural weight downloads.

#### 4. `src/geometry.py`
- **Automatic UTM Projection**: Given any longitude and latitude, calculates the exact WGS84 UTM Zone ($EPSG = 32600 + zone$ North, $32700 + zone$ South).
- **Inscribed Crown Ellipses**: Models crowns as rounded ellipses rather than rigid rectangles, eliminating the standard $\sim 21.5\%$ bounding-box overestimation error.
- **Spatial Overlap Validation**: Checks if an uploaded AOI is `FULL_INSIDE`, `COVERS_IMAGE`, `PARTIAL_OVERLAP`, or `NO_OVERLAP` relative to the raster.
- **Canopy Dissolution**: Uses `shapely.ops.unary_union` to merge overlapping crowns into a single continuous polygon footprint.

#### 5. `src/metrics.py`
- **Canopy Metrics**: Calculates Tree Count, Raw Box Area ($m^2$), Dissolved Canopy Area ($m^2$ & hectares), Overlap Redundancy ($m^2$ & %), Canopy Cover Percentage, and Trees per Hectare.
- **Distribution Analysis**: Computes mean, median, min, max, and standard deviation of crown diameters.
- **Ethical & Methodological Disclosure**: Explicitly details all assumptions, sensor limits, and provides the mandatory carbon offset legal disclaimer.

#### 6. `src/visualization.py`
- **Image Annotations**: Color-coded bounding boxes and crown centroids (Spring Green for high confidence, Gold for medium, Tomato Red for low).
- **Interactive Folium Maps**: Satellite basemap (Esri World Imagery) overlayed with AOI vector boundary, dissolved canopy cover footprint, and individual crown popups.
- **Plotly Visualizations**: Interactive histograms for crown diameters and comparative bar charts demonstrating the gap between naive box sums and true dissolved canopy.

---

## 🧮 Mathematical Formulations

### 1. Ground Sample Distance (GSD)
For projected GeoTIFFs:
$$\text{GSD}_x = |a|, \quad \text{GSD}_y = |e|$$
For geographic GeoTIFFs at latitude $\phi$:
$$\text{GSD}_x = |\Delta \lambda| \cdot \left(111412.84 \cos\phi - 93.5 \cos(3\phi)\right)$$
$$\text{GSD}_y = |\Delta \phi| \cdot \left(111132.95 - 559.82 \cos(2\phi) + 1.18 \cos(4\phi)\right)$$

### 2. Dissolved Canopy Area vs. Naive Summation
Given $N$ detected crown polygons $\{P_1, P_2, \dots, P_N\}$:
$$\text{Area}_{\text{raw}} = \sum_{i=1}^N \text{Area}(P_i)$$
$$\text{Area}_{\text{dissolved}} = \text{Area}\left(\bigcup_{i=1}^N P_i\right)$$
$$\text{Redundancy}_{\text{overlap}} = \text{Area}_{\text{raw}} - \text{Area}_{\text{dissolved}}$$

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.11 (recommended)
- Windows / Linux / macOS

### 1. Environment Setup
```bash
# Create and activate Python 3.11 virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Running Unit Tests
```bash
# Run lightweight unit tests (no model weights or large imagery required)
python -m unittest discover tests
```

### 3. Running the Streamlit Application
```bash
streamlit run app.py
```

---

## ⚠️ Uncertainty, Limitations & Regulatory Disclaimer

1. **Sub-Canopy Occlusion**: Optical nadir sensors cannot detect understory trees occluded beneath dominant canopy.
2. **Dense Canopy Clumping**: In interlocking closed canopies, detector boundaries may group multi-stem crowns into single detections.
3. **Sensor Limits**: Trees with crown diameters smaller than $3 \times \text{GSD}$ cannot be reliably resolved.
4. **Carbon Offsets & Biomass**: This software estimates 2D surface canopy cover only. **It does not compute aboveground biomass (AGB) or verified carbon credits.** Carbon credit verification requires calibrated ground inventory plots, wood density sampling, LiDAR height profiles, and certified audit methodologies.
