"""
Generates real sample datasets in demo_data/ for instant one-click demo testing.
Creates:
- demo_data/sample_forest_utm17n.tif (Georeferenced GeoTIFF in EPSG:32617 with 0.10m GSD)
- demo_data/sample_boundary.kml (Matching KML boundary intersecting the imagery)
- demo_data/sample_aerial_photo.png (Standard PNG aerial imagery with known 0.10m GSD)
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image
import deepforest

def generate_demo_assets():
    demo_dir = Path("demo_data")
    demo_dir.mkdir(parents=True, exist_ok=True)

    # 1. Source real aerial imagery from DeepForest package
    deepforest_dir = Path(deepforest.__file__).parent
    osbs_path = deepforest_dir / "data" / "OSBS_029.png"
    if not osbs_path.exists():
        raise FileNotFoundError(f"DeepForest test image not found at {osbs_path}")

    # Copy standard PNG
    sample_png = demo_dir / "sample_aerial_photo.png"
    img_pil = Image.open(osbs_path)
    img_pil.save(sample_png)
    print(f"[OK] Saved {sample_png} ({img_pil.size[0]}x{img_pil.size[1]} px)")

    # 2. Create Georeferenced GeoTIFF (WGS 84 / UTM Zone 17N, EPSG:32617)
    sample_tif = demo_dir / "sample_forest_utm17n.tif"
    img_np = np.array(img_pil)
    h, w, c = img_np.shape

    # Origin at Ordway-Swisher Biological Station, FL (UTM 17N)
    origin_x = 403800.0
    origin_y = 3284900.0
    gsd_m = 0.10
    transform = from_origin(origin_x, origin_y, gsd_m, gsd_m)

    with rasterio.open(
        sample_tif,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=img_np.dtype,
        crs="EPSG:32617",
        transform=transform,
    ) as dst:
        for band in range(c):
            dst.write(img_np[:, :, band], band + 1)
    print(f"[OK] Saved {sample_tif} (EPSG:32617, GSD: {gsd_m} m/px)")

    # 3. Create Matching KML Boundary Polygon
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:32617", "EPSG:4326", always_xy=True)

    # Sub-polygon covering central forest section
    min_x, max_x = origin_x + 6.0, origin_x + 32.0
    min_y, max_y = origin_y - 34.0, origin_y - 8.0

    coords_utm = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y)
    ]
    coords_wgs84 = [transformer.transform(x, y) for x, y in coords_utm]
    kml_coords_str = " ".join(f"{lon},{lat},0" for lon, lat in coords_wgs84)

    sample_kml = demo_dir / "sample_boundary.kml"
    kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Sample Forest AOI Boundary</name>
    <description>Ordway-Swisher Biological Station Forest Compartment A</description>
    <Placemark>
      <name>Compartment A</name>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              {kml_coords_str}
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>
"""
    with open(sample_kml, "w", encoding="utf-8") as f:
        f.write(kml_content)
    print(f"[OK] Saved {sample_kml}")

if __name__ == "__main__":
    generate_demo_assets()
