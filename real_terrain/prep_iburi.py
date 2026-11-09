"""
Build the analysis grid for the 2018 Hokkaido Eastern Iburi earthquake area.

Output: data/processed/iburi_grid.npz with, on a 120 m UTM grid (EPSG:3100):
  - terrain features (from Copernicus DEM, pre-event, computed at 30 m then aggregated)
  - access features (distance to roads by class, built-up density)
  - landslide fraction per cell and the "true" label z (full inventory, used as truth)
  - study mask (convex hull of the inventory, i.e. the mapped area)

Land cover is deliberately NOT used as a terrain feature: WorldCover 2021 is post-event and
shows landslide scars as bare/grass, which would leak the answer. Only its built-up class is
used, as a proxy for where people live (an observation-process feature).

Usage: python prep_iburi.py <raw_data_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy import ndimage
from shapely.geometry import LineString

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
CRS = "EPSG:3100"          # JGD2000 / UTM zone 54N (same CRS as the inventory)
FINE, COARSE = 30.0, 120.0
F = int(COARSE // FINE)    # 4x4 fine cells per coarse cell

# ------------------------------------------------------------------ inventory & grid
inv = gpd.read_file(RAW / "iburi_inventory/Iburi landslide dataset/landslide_inventory/landslide_inventory.shp")
inv = inv.to_crs(CRS)
hull = inv.geometry.union_all().convex_hull
x0, y0, x1, y1 = hull.buffer(1500).bounds
x0, y0 = np.floor(x0 / COARSE) * COARSE, np.floor(y0 / COARSE) * COARSE
x1, y1 = np.ceil(x1 / COARSE) * COARSE, np.ceil(y1 / COARSE) * COARSE
Wc, Hc = int((x1 - x0) / COARSE), int((y1 - y0) / COARSE)
Wf, Hf = Wc * F, Hc * F
tf_fine = from_origin(x0, y1, FINE, FINE)
tf_coarse = from_origin(x0, y1, COARSE, COARSE)
print(f"grid: {Hc} x {Wc} coarse cells ({COARSE:.0f} m), fine {Hf} x {Wf} ({FINE:.0f} m)")


def block(a, fn=np.mean):
    return fn(a.reshape(Hc, F, Wc, F), axis=(1, 3))


# ------------------------------------------------------------------ DEM -> 30 m UTM
srcs = [rasterio.open(p) for p in sorted((RAW / "dem").glob("*.tif"))]
mosaic, mtf = merge(srcs)
dem = np.zeros((Hf, Wf), np.float32)
reproject(mosaic[0], dem, src_transform=mtf, src_crs=srcs[0].crs,
          dst_transform=tf_fine, dst_crs=CRS, resampling=Resampling.bilinear)
dem = ndimage.gaussian_filter(dem, 0.7)           # light smoothing of DSM noise

gy, gx = np.gradient(dem, FINE)
gy = -gy                                          # rows go north->south
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
aspect = np.arctan2(-gx, gy)                      # 0 = north-facing slope (downhill direction)
# curvature: Laplacian of smoothed DEM (negative = concave hollow)
curv = ndimage.laplace(ndimage.gaussian_filter(dem, 1.0)) / FINE ** 2


def tpi(r_cells):
    return dem - ndimage.uniform_filter(dem, size=2 * r_cells + 1)


def rel_height(r_cells):
    """height above local minimum within a window (crude proxy for height above drainage)"""
    return dem - ndimage.minimum_filter(dem, size=2 * r_cells + 1)


terrain_fine = {
    "elev": dem,
    "slope_mean": slope,
    "aspect_sin": np.sin(aspect),
    "aspect_cos": np.cos(aspect),
    "curvature": curv,
    "tpi_150m": tpi(5),
    "tpi_600m": tpi(20),
    "rel_height_450m": rel_height(15),
}
terrain = {k: block(v) for k, v in terrain_fine.items()}
terrain["slope_max"] = block(slope, np.max)
terrain["slope_std"] = block(slope, np.std)
terrain["relief"] = block(dem, np.max) - block(dem, np.min)

# ------------------------------------------------------------------ landslides -> fraction per cell
ls_fine = features.rasterize(((g, 1) for g in inv.geometry), out_shape=(Hf, Wf),
                             transform=tf_fine, fill=0, dtype="uint8")
# 30 m raster underestimates small polygons; also mark the cell holding each polygon's centroid
ls_frac = block(ls_fine.astype(np.float32))
cent = inv.geometry.representative_point()
ci = ((y1 - cent.y) // COARSE).astype(int).clip(0, Hc - 1)
cj = ((cent.x - x0) // COARSE).astype(int).clip(0, Wc - 1)
has_centroid = np.zeros((Hc, Wc), bool); has_centroid[ci, cj] = True
z = ((ls_frac >= 0.10) | has_centroid).astype(np.int8)

mask = features.rasterize([(hull, 1)], out_shape=(Hc, Wc), transform=tf_coarse,
                          fill=0, dtype="uint8").astype(bool)

# ------------------------------------------------------------------ roads -> distances
osm = json.loads((RAW / "osm/roads.json").read_text())
MAJOR = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
         "secondary", "secondary_link", "tertiary", "tertiary_link"}
MINOR = {"unclassified", "residential", "service", "living_street"}
TRACK = {"track"}
lines = {"major": [], "minor": [], "track": []}
for el in osm["elements"]:
    hw = el.get("tags", {}).get("highway")
    geom = el.get("geometry")
    if not geom or len(geom) < 2:
        continue
    cls = "major" if hw in MAJOR else "minor" if hw in MINOR else "track" if hw in TRACK else None
    if cls:
        lines[cls].append(LineString([(p["lon"], p["lat"]) for p in geom]))
dist = {}
for cls, ls in lines.items():
    gs = gpd.GeoSeries(ls, crs="EPSG:4326").to_crs(CRS)
    r = features.rasterize(((g, 1) for g in gs), out_shape=(Hf, Wf), transform=tf_fine,
                           fill=0, dtype="uint8", all_touched=True)
    dist[cls] = block(ndimage.distance_transform_edt(r == 0) * FINE) / 1000.0   # km
    print(f"{cls:6s} roads: {len(ls):5d} ways  median dist in study area = {np.median(dist[cls][mask]):.2f} km")
dist["public"] = np.minimum(dist["major"], dist["minor"])      # roads a car/agency would use

# ------------------------------------------------------------------ built-up density (WorldCover class 50)
with rasterio.open(RAW / "landcover/ESA_WorldCover_10m_2021_v200_N42E141_Map.tif") as src:
    lc = np.zeros((Hf, Wf), np.uint8)
    reproject(rasterio.band(src, 1), lc, dst_transform=tf_fine, dst_crs=CRS,
              resampling=Resampling.mode)
built = block((lc == 50).astype(np.float32))
built_density = ndimage.gaussian_filter(built, 2000 / COARSE)   # ~2 km neighbourhood

np.savez_compressed(
    OUT / "iburi_grid.npz",
    **{f"terrain__{k}": v.astype(np.float32) for k, v in terrain.items()},
    **{f"dist__{k}": v.astype(np.float32) for k, v in dist.items()},
    built_density=built_density.astype(np.float32),
    ls_frac=ls_frac.astype(np.float32), z=z, mask=mask,
    transform=np.array(tf_coarse)[:6], crs=CRS, x0=x0, y1=y1, cell=COARSE)

m = mask
print(f"study cells: {m.sum()}  landslide cells: {z[m].sum()} ({z[m].mean():.1%})  "
      f"landslides outside mask: {z[~m].sum()}")
print("slope_mean in landslide vs other cells: "
      f"{terrain['slope_mean'][m & (z == 1)].mean():.1f} vs {terrain['slope_mean'][m & (z == 0)].mean():.1f} deg")
for cls in ["major", "public", "track"]:
    print(f"dist {cls}: landslide cells {np.median(dist[cls][m & (z == 1)]):.2f} km vs others "
          f"{np.median(dist[cls][m & (z == 0)]):.2f} km")
