"""
Build an analysis grid for ANY earthquake-triggered landslide inventory (generalises prep_iburi.py).

Example (Iburi, reproduces prep_iburi.py):
  python prep_generic.py --inventory RAW/iburi_inventory/.../landslide_inventory.shp \
      --dem-dir RAW/dem --roads RAW/osm/roads.json --landcover RAW/landcover/*.tif \
      --crs EPSG:3100 --cell 120 --epicenters "42.686,141.929" --out ../data/processed/iburi_grid_generic.npz

Mask = mapped extent (if --extent given; otherwise convex hull of the inventory), minus obscured areas
(--obscured), optionally clipped to a lon/lat window (--window S,W,N,E).
Terrain from the DEM (pre-event); access = distance to roads by class + built-up density (WorldCover
class 50 only; the other classes are post-event and would leak landslide scars).
"""
import argparse
import json
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
from shapely.geometry import LineString, box

MAJOR = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
         "secondary", "secondary_link", "tertiary", "tertiary_link"}
MINOR = {"unclassified", "residential", "service", "living_street"}
TRACK = {"track"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--extent", default=None)
    ap.add_argument("--obscured", default=None)
    ap.add_argument("--window", default=None, help="S,W,N,E in degrees")
    ap.add_argument("--dem-dir", required=True)
    ap.add_argument("--roads", nargs="+", required=True)
    ap.add_argument("--landcover", nargs="+", required=True)
    ap.add_argument("--crs", required=True)
    ap.add_argument("--cell", type=float, default=120.0)
    ap.add_argument("--fine", type=float, default=30.0)
    ap.add_argument("--epicenters", required=True, help='"lat,lon;lat,lon"')
    ap.add_argument("--min-frac", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    CRS, COARSE, FINE = a.crs, a.cell, a.fine
    F = int(round(COARSE / FINE))

    inv = gpd.read_file(a.inventory).to_crs(CRS)
    inv = inv[inv.geometry.notna() & ~inv.geometry.is_empty]
    if not inv.geometry.is_valid.all():                     # some published polygons have broken rings
        print(f"repairing {int((~inv.geometry.is_valid).sum())} invalid geometries")
        inv["geometry"] = inv.geometry.make_valid()
    region = (gpd.read_file(a.extent).to_crs(CRS).geometry.union_all() if a.extent
              else inv.geometry.union_all().convex_hull)
    if a.obscured:
        region = region.difference(gpd.read_file(a.obscured).to_crs(CRS).geometry.union_all())
    if a.window:
        S, W, N, E = map(float, a.window.split(","))
        win = gpd.GeoSeries([box(W, S, E, N)], crs="EPSG:4326").to_crs(CRS).iloc[0]
        region = region.intersection(win)
        inv = inv[inv.intersects(win)]
    x0, y0, x1, y1 = region.buffer(1500).bounds
    x0, y0 = np.floor(x0 / COARSE) * COARSE, np.floor(y0 / COARSE) * COARSE
    x1, y1 = np.ceil(x1 / COARSE) * COARSE, np.ceil(y1 / COARSE) * COARSE
    Wc, Hc = int((x1 - x0) / COARSE), int((y1 - y0) / COARSE)
    Wf, Hf = Wc * F, Hc * F
    tf_f, tf_c = from_origin(x0, y1, FINE, FINE), from_origin(x0, y1, COARSE, COARSE)
    print(f"grid {Hc} x {Wc} cells of {COARSE:.0f} m; {len(inv)} landslide polygons")
    block = lambda arr, fn=np.mean: fn(arr.reshape(Hc, F, Wc, F), axis=(1, 3))

    # DEM
    srcs = [rasterio.open(p) for p in sorted(Path(a.dem_dir).glob("*.tif"))]
    mos, mtf = merge(srcs)
    dem = np.zeros((Hf, Wf), np.float32)
    reproject(mos[0], dem, src_transform=mtf, src_crs=srcs[0].crs, dst_transform=tf_f, dst_crs=CRS,
              resampling=Resampling.bilinear)
    dem = ndimage.gaussian_filter(dem, 0.7)
    gy, gx = np.gradient(dem, FINE); gy = -gy
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.arctan2(-gx, gy)
    curv = ndimage.laplace(ndimage.gaussian_filter(dem, 1.0)) / FINE ** 2
    r = lambda metres: max(1, int(round(metres / FINE)))
    tpi = lambda rc: dem - ndimage.uniform_filter(dem, size=2 * rc + 1)
    terrain = {k: block(v) for k, v in {
        "elev": dem, "slope_mean": slope, "aspect_sin": np.sin(aspect), "aspect_cos": np.cos(aspect),
        "curvature": curv, "tpi_150m": tpi(r(150)), "tpi_600m": tpi(r(600)),
        "rel_height_450m": dem - ndimage.minimum_filter(dem, size=2 * r(450) + 1)}.items()}
    terrain["slope_max"] = block(slope, np.max)
    terrain["slope_std"] = block(slope, np.std)
    terrain["relief"] = block(dem, np.max) - block(dem, np.min)

    # epicentral distances
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    xs = x0 + (np.arange(Wc) + 0.5) * COARSE; ys = y1 - (np.arange(Hc) + 0.5) * COARSE
    X, Y = np.meshgrid(xs, ys)
    epis = [tuple(map(float, e.split(","))) for e in a.epicenters.split(";")]
    for i, (lat, lon) in enumerate(epis):
        ex, ey = tr.transform(lon, lat)
        terrain["epi_dist_km" if i == 0 else f"epi{i + 1}_dist_km"] = np.hypot(X - ex, Y - ey) / 1000.0

    # landslides
    ls_f = features.rasterize(((g, 1) for g in inv.geometry), out_shape=(Hf, Wf), transform=tf_f,
                              fill=0, dtype="uint8")
    ls_frac = block(ls_f.astype(np.float32))
    pts = inv.geometry.representative_point()
    ci = ((y1 - pts.y) // COARSE).astype(int).clip(0, Hc - 1)
    cj = ((pts.x - x0) // COARSE).astype(int).clip(0, Wc - 1)
    cen = np.zeros((Hc, Wc), bool); cen[ci, cj] = True
    z = ((ls_frac >= a.min_frac) | cen).astype(np.int8)
    # mask: cell centre inside region
    mask = features.rasterize([(region, 1)], out_shape=(Hc, Wc), transform=tf_c, fill=0,
                              dtype="uint8").astype(bool)

    # roads
    lines = {"major": [], "minor": [], "track": []}
    for rp in a.roads:
        for el in json.loads(Path(rp).read_text())["elements"]:
            hw = el.get("tags", {}).get("highway"); geom = el.get("geometry")
            if not geom or len(geom) < 2:
                continue
            cls = "major" if hw in MAJOR else "minor" if hw in MINOR else "track" if hw in TRACK else None
            if cls:
                lines[cls].append(LineString([(p["lon"], p["lat"]) for p in geom]))
    dist = {}
    for cls, ls in lines.items():
        rr = np.zeros((Hf, Wf), np.uint8)
        if ls:
            gs = gpd.GeoSeries(ls, crs="EPSG:4326").to_crs(CRS)
            rr = features.rasterize(((g, 1) for g in gs), out_shape=(Hf, Wf), transform=tf_f, fill=0,
                                    dtype="uint8", all_touched=True)
        dist[cls] = block(ndimage.distance_transform_edt(rr == 0) * FINE) / 1000.0 if rr.any() \
            else np.full((Hc, Wc), 50.0, np.float32)
        print(f"  {cls:6s} roads: {len(ls):6d} ways, median distance in mask {np.median(dist[cls][mask]):.2f} km")
    dist["public"] = np.minimum(dist["major"], dist["minor"])

    # built-up density
    built_f = np.zeros((Hf, Wf), np.float32)
    for lp in a.landcover:
        with rasterio.open(lp) as src:
            lc = np.zeros((Hf, Wf), np.uint8)
            reproject(rasterio.band(src, 1), lc, dst_transform=tf_f, dst_crs=CRS, resampling=Resampling.mode)
            built_f = np.maximum(built_f, (lc == 50).astype(np.float32))
    built_density = ndimage.gaussian_filter(block(built_f), 2000 / COARSE)

    np.savez_compressed(a.out, **{f"terrain__{k}": v.astype(np.float32) for k, v in terrain.items()},
                        **{f"dist__{k}": v.astype(np.float32) for k, v in dist.items()},
                        built_density=built_density.astype(np.float32), ls_frac=ls_frac.astype(np.float32),
                        z=z, mask=mask, transform=np.array(tf_c)[:6], crs=CRS, x0=x0, y1=y1, cell=COARSE)
    print(f"mask cells {mask.sum()}, landslide cells {z[mask].sum()} ({z[mask].mean():.1%}); "
          f"slope landslide vs other {terrain['slope_mean'][mask & (z == 1)].mean():.1f} vs "
          f"{terrain['slope_mean'][mask & (z == 0)].mean():.1f} deg; median public-road distance landslide vs other "
          f"{np.median(dist['public'][mask & (z == 1)]):.2f} vs {np.median(dist['public'][mask & (z == 0)]):.2f} km")


if __name__ == "__main__":
    main()
