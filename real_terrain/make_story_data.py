"""
Build the data file for the 3D story (docs/story-data.js) from the real Iburi data and the paper's draw 0, scenario A.

Contents, for a 24 x 24 km window around the densest landslides (EPSG:3100):
  terrain   Copernicus GLO-30 elevation, resampled to a 321 x 321 grid
  scars     every Zhang et al. (2019) landslide polygon in the window: centroid, area, elongation,
            distance to a public road, whether the simulated records contain its cell (as in paper Fig. 3b),
            and its percentile under the standard and occupancy models where the cell is in a held-out test block
  roads     OpenStreetMap motorway to tertiary roads, simplified
  heat      standard and occupancy model percentiles on the 120 m grid, test blocks only (as in paper Fig. 3d, 3f)

Needs the raw downloads (data/raw, see data/manifests/iburi_raw.md) plus: pip install pyshp tifffile imagecodecs
Usage: python make_story_data.py ../data/raw ../docs/story-data.js
"""
import ast
import base64
import json
import sys
from pathlib import Path

import numpy as np
import shapefile
import tifffile
from pyproj import Transformer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "toy_simulation"))
sys.path.insert(0, str(HERE))
from run_iburi import load_grid, simulate_inventories  # noqa: E402

KM = 24
X0, Y0 = 568182.0, 4723766.7          # south-west corner, chosen to hold the most landslides
NT = 321                              # terrain samples per side
EPI_LONLAT = (141.929, 42.686)
ROAD_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary"}
STD, OURS = "Standard (terrain + access)", "Occupancy + monotone + anchor (FULL)"


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def rank01(a):
    r = np.empty(len(a)); r[np.argsort(a)] = np.linspace(0, 1, len(a)); return r


def main(raw, out):
    raw = Path(raw)
    to_ll = Transformer.from_crs("EPSG:3100", "EPSG:4326", always_xy=True)
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:3100", always_xy=True)
    span = KM * 1000.0
    uv = lambda x, y: ((x - X0) / span, (Y0 + span - y) / span)

    # terrain: bilinear samples of the two 1-degree DEM tiles
    tiles = {lon: tifffile.imread(raw / f"dem/Copernicus_DSM_COG_10_N42_00_E{lon}_00_DEM.tif") for lon in (141, 142)}
    t = (np.arange(NT) + 0.5) / NT
    gx, gy = np.meshgrid(X0 + t * span, Y0 + span - t * span)
    lon, lat = to_ll.transform(gx.ravel(), gy.ravel())
    lon, lat = np.asarray(lon), np.asarray(lat)
    elev = np.empty(lon.size)
    for t0, dem in tiles.items():
        s = (lon >= t0) & (lon < t0 + 1)
        c, r = (lon[s] - t0) * 3600, (43 - lat[s]) * 3600
        i, j = np.floor(c).astype(int), np.floor(r).astype(int)
        fx, fy = c - i, r - j
        i1, j1 = np.minimum(i + 1, 3599), np.minimum(j + 1, 3599)
        elev[s] = (dem[j, i] * (1 - fx) * (1 - fy) + dem[j, i1] * fx * (1 - fy)
                   + dem[j1, i] * (1 - fx) * fy + dem[j1, i1] * fx * fy)
    elev = np.clip(np.round(elev.reshape(NT, NT)), 0, 32767).astype("<i2")

    # paper's grid, simulated records (draw 0) and model percentiles in the test blocks
    d, m, terrain, dist = load_grid(HERE.parent / "data/processed/iburi_grid.npz")
    y, _, _, _ = simulate_inventories(d["z"], m, terrain, dist, d["built_density"], np.random.default_rng(0))
    rec = y.max(0).astype(bool)
    P = np.load(HERE / "results_v3/seed0_A.npz")
    te = np.zeros_like(m); te[m] = P["test"]
    pct = {}
    for key, name in (("std", STD), ("ours", OURS)):
        g = np.full(m.shape, np.nan); g[te] = rank01(P[name]); pct[key] = g
    gx0, gy1, cell = float(d["x0"]), float(d["y1"]), float(d["cell"])
    ci0, ri0, nh = int(round((X0 - gx0) / cell)), int(round((gy1 - (Y0 + span)) / cell)), int(span / cell)
    win = lambda a: a[ri0:ri0 + nh, ci0:ci0 + nh]
    heat = {k: np.where(np.isnan(win(g)), 0, 1 + np.round(win(g) * 254)).astype(np.uint8) for k, g in pct.items()}
    cell_of = lambda x, yy: (int((gy1 - yy) // cell), int((x - gx0) // cell))

    # landslides
    sf = shapefile.Reader(str(raw / "iburi_inventory/Iburi landslide dataset/landslide_inventory/landslide_inventory.shp"),
                          encoding="cp932")
    scars = []
    for shp in sf.shapes():
        p = np.asarray(shp.points)
        cx, cy = p[:, 0].mean(), p[:, 1].mean()
        u, v = uv(cx, cy)
        if not (0 <= u < 1 and 0 <= v < 1):
            continue
        q = p - p.mean(0)
        area = abs(sum(0.5 * (np.dot(r[:-1, 0], r[1:, 1]) - np.dot(r[1:, 0], r[:-1, 1]))
                       for r in np.split(q, shp.parts[1:]) if len(r) > 2))
        ev = np.linalg.eigvalsh(np.cov((p - p.mean(0)).T))
        elong = float(np.clip(np.sqrt(ev[1] / max(ev[0], 1e-9)), 1, 5))
        r, c = cell_of(cx, cy)
        road = float(dist["public"][r, c] * 1000)
        ps, po = pct["std"][r, c], pct["ours"][r, c]
        scars.append([round(u, 5), round(v, 5), int(round(area)), round(elong, 2), int(round(road)), int(rec[r, c]),
                      -1 if np.isnan(ps) else int(round(ps * 100)), -1 if np.isnan(po) else int(round(po * 100))])

    # roads
    roads = []
    for e in json.load(open(raw / "osm/roads.json"))["elements"]:
        if e.get("tags", {}).get("highway") not in ROAD_CLASSES:
            continue
        g = e["geometry"]; g = ast.literal_eval(g) if isinstance(g, str) else g
        xs, ys = to_utm.transform([q["lon"] for q in g], [q["lat"] for q in g])
        pts = [uv(a, b) for a, b in zip(xs, ys)]
        keep, last = [], None
        for q in pts:
            if last is None or np.hypot(q[0] - last[0], q[1] - last[1]) > 0.002:
                keep.append(q); last = q
        if pts and keep[-1] != pts[-1]:
            keep.append(pts[-1])
        inside = [q for q in keep if -0.02 <= q[0] <= 1.02 and -0.02 <= q[1] <= 1.02]
        if len(inside) >= 2:
            roads.append([[round(a, 4), round(b, 4)] for a, b in keep])

    ex, ey = to_utm.transform(*EPI_LONLAT)
    data = dict(km=KM, nt=NT, terrain=b64(elev), hg=nh, heatStd=b64(heat["std"]), heatOurs=b64(heat["ours"]),
                epi=[round(c, 4) for c in uv(ex, ey)], nInventory=len(sf), scars=scars, roads=roads,
                source="Zhang et al. (2019) inventory; Copernicus GLO-30; OpenStreetMap; Omar & Ugaas (2026) draw 0, scenario A")
    Path(out).write_text("window.STORY_DATA=" + json.dumps(data, separators=(",", ":")) + ";\n")
    s = np.array(scars)
    print(f"{len(scars)} landslides in window, {int(s[:, 5].sum())} recorded ({s[:, 5].mean():.0%}); "
          f"{(s[:, 6] >= 0).sum()} in test blocks; {len(roads)} roads; epicentre uv {data['epi']}; "
          f"{Path(out).stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
