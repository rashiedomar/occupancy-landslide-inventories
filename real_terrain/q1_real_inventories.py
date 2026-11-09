"""
Q1 step 1: put the REAL independent Gorkha inventories (USGS Ground Failure Database v4) on the
Gorkha grid and characterise their real-world biases relative to the reference (Roback source areas).

Writes data/processed/gorkha_real_obs.npz with one 0/1 raster per real inventory and its coverage
(convex hull of that inventory's features inside the window), plus a bias report.
Usage: python q1_real_inventories.py <gorkha_window_polygons.gpkg>
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio import features
from rasterio.transform import Affine

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run_iburi import load_grid  # noqa: E402

GRID = HERE.parent / "data/processed/gorkha_grid.npz"
OBS = {"Gnyawali (2016)": "Gnyawali and others (2016)",
       "Valagussa (2021)": "Valagussa and Others (2021)",
       "Zhang (2016)": "Zhang and others (2016)"}


def main(gpkg):
    d, m, terrain, dist = load_grid(GRID)
    H, W = m.shape
    tf = Affine(*d["transform"])
    crs = str(d["crs"])
    z = d["z"].astype(bool)
    pol = gpd.read_file(gpkg).to_crs(crs)
    pol["geometry"] = pol.geometry.make_valid()
    rasters, covers, report = {}, {}, {}
    for short, key in OBS.items():
        g = pol[pol["inventory_"] == key]
        r = features.rasterize(((geom, 1) for geom in g.geometry), out_shape=(H, W), transform=tf,
                               fill=0, dtype="uint8", all_touched=False)
        pts = g.geometry.representative_point()
        ci = np.clip(((d["y1"] - pts.y) // d["cell"]).astype(int), 0, H - 1)
        cj = np.clip(((pts.x - d["x0"]) // d["cell"]).astype(int), 0, W - 1)
        r[ci, cj] = 1
        hull = g.geometry.union_all().convex_hull
        cov = features.rasterize([(hull, 1)], out_shape=(H, W), transform=tf, fill=0, dtype="uint8").astype(bool)
        y = r.astype(bool) & m
        rasters[short], covers[short] = y, cov & m
        cm = cov & m
        recall = (y & z & cm).sum() / max((z & cm).sum(), 1)
        extra = (y & ~z & cm).sum() / max((y & cm).sum(), 1)
        # real bias: recall of reference landslides by quartile of road distance / slope / aspect
        dp = dist["public"]; sl = terrain["slope_mean"]; asp = terrain["aspect_cos"]
        def by_q(v):
            qs = np.quantile(v[z & cm], [0.25, 0.5, 0.75])
            b = np.digitize(v, qs)
            return [float((y & z & cm & (b == k)).sum() / max((z & cm & (b == k)).sum(), 1)) for k in range(4)]
        report[short] = dict(n_cells=int(y.sum()), coverage_frac=float(cm.sum() / m.sum()),
                             recall_of_reference=float(recall), share_not_in_reference=float(extra),
                             recall_by_road_distance_quartile=by_q(dp),
                             recall_by_slope_quartile=by_q(sl),
                             recall_by_northness_quartile=by_q(asp))
        print(f"{short:18s} cells {y.sum():5d} | coverage {cm.sum() / m.sum():.0%} | catches {recall:.0%} of reference "
              f"landslide cells | {extra:.0%} of its cells not in reference")
        print(f"{'':18s} catch rate by road-distance quartile (near->far): "
              + " ".join(f"{v:.0%}" for v in report[short]['recall_by_road_distance_quartile'])
              + " | by slope (gentle->steep): " + " ".join(f"{v:.0%}" for v in report[short]['recall_by_slope_quartile'])
              + " | by northness (S->N facing): " + " ".join(f"{v:.0%}" for v in report[short]['recall_by_northness_quartile']))
    union = np.any([rasters[k] for k in rasters], 0)
    print(f"union of real inventories catches {(union & z & m).sum() / (z & m).sum():.0%} of reference cells; "
          f"{(union & ~z & m).sum() / (union & m).sum():.0%} of union cells not in reference")
    np.savez_compressed(HERE.parent / "data/processed/gorkha_real_obs.npz",
                        names=np.array(list(rasters)), y=np.stack(list(rasters.values())).astype(np.int8),
                        cover=np.stack(list(covers.values())))
    (HERE / "results_q1").mkdir(exist_ok=True)
    (HERE / "results_q1/real_inventory_bias.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
