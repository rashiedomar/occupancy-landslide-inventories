"""
Q1b: a REAL report-based catalog as an observer (NASA Global Landslide Catalog / COOLR: news, disaster
databases, citizen reports) -- the road/population-biased kind of record the method was designed for.

Observers: the report catalog (+ any real satellite inventories given) + the anchor area.
Truth: the reference inventory on the grid. Scored like the other experiments.
First prints the catalog's own bias: share of report points by road-distance quartile vs the truth.

Usage: python q4_reports.py <grid.npz> <reports.geojson> <name> [<real_obs.npz>]
"""
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from rasterio.transform import Affine

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "toy_simulation"))
sys.path.insert(0, str(HERE))
from models import BinaryClassifier, MonotoneDetection, OccupancyNet, train  # noqa: E402
from q1_run_real import ACC, ACC_SIGNS, VIS, metrics  # noqa: E402
from run_iburi import _z, block_split, load_grid  # noqa: E402
from run_iburi_v3 import pick_anchor  # noqa: E402

torch.set_num_threads(2)


def rasterise_points(gdf, d, m):
    H, W = m.shape
    pts = gdf.to_crs(str(d["crs"])).geometry
    pts = pts[~pts.is_empty]
    r = ((d["y1"] - pts.y) // d["cell"]).astype(int)
    c = ((pts.x - d["x0"]) // d["cell"]).astype(int)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    y = np.zeros((H, W), bool)
    y[r[ok], c[ok]] = True
    return y & m, int(ok.sum())


def main(grid_path, reports_path, name, obs_path=None, seeds=5, restarts=3, steps=3000):
    grid = load_grid(grid_path)
    d, m, terrain, dist = grid
    z = d["z"].astype(np.int8)
    rep = gpd.read_file(reports_path)
    y_rep, n_in = rasterise_points(rep, d, m)
    dp = dist["public"]
    qs = np.quantile(dp[m & (z == 1)], [0.25, 0.5, 0.75])
    b = np.digitize(dp, qs)
    share_rep = [float((y_rep & (b == k)).sum() / max(y_rep.sum(), 1)) for k in range(4)]
    print(f"{name}: {len(rep)} report features, {n_in} inside grid, {int(y_rep.sum())} cells; "
          f"{int((y_rep & (z == 1)).sum())} on reference landslide cells")
    print(f"  share of report cells by road-distance quartile of true landslides (near->far): "
          + " ".join(f"{v:.0%}" for v in share_rep) + "  (unbiased = 25% each)")
    ys, covs = [y_rep], [m.copy()]
    if obs_path:
        o = np.load(obs_path)
        ys += [v.astype(bool) for v in o["y"]]; covs += [v for v in o["cover"]]
    K = len(ys)
    res = {"_catalog": dict(n_features=len(rep), n_cells=int(y_rep.sum()), share_by_road_quartile=share_rep)}
    for seed in range(seeds):
        t0 = time.time()
        rng = np.random.default_rng(seed)
        split = block_split(m, rng)
        anchor = pick_anchor(split, m, np.random.default_rng(10_000 + seed))
        Xt = np.stack([_z(terrain[k], m)[m] for k in sorted(terrain)], -1).astype(np.float32)
        acc = {"log_d_major": np.log1p(dist["major"]), "log_d_public": np.log1p(dist["public"]),
               "log_d_track": np.log1p(dist["track"]), "built": d["built_density"]}
        Xa = np.stack([_z(acc[k], m)[m] for k in ACC], -1).astype(np.float32)
        Xv = np.stack([_z(terrain[k], m)[m] for k in VIS], -1).astype(np.float32)
        Xo = np.concatenate([Xa, Xv, Xv], 1)
        signs = ACC_SIGNS + [+1] * len(VIS) + [-1] * len(VIS)
        Y = np.concatenate([np.stack([v[m] for v in ys], 1), ((z == 1) & anchor)[m][:, None]], 1).astype(np.int8)
        M = np.concatenate([np.stack([v[m] for v in covs], 1), anchor[m][:, None]], 1)
        sp = split[m]; tr, va, te = sp == 0, sp == 1, sp == 2
        D = dict(z=z[m], cons=z[m], slope=terrain["slope_mean"][m], d_pub=dp[m], te=te)
        T = lambda a: torch.tensor(a[tr]); V = lambda a: torch.tensor(a[va])
        y_any = Y.max(1); nt = Xt.shape[1]
        X_all = np.concatenate([Xt, Xa], 1); X_neu = X_all[te].copy(); X_neu[:, nt:] = 0.0

        def ens(make, loss_on):
            ms = []
            for r in range(restarts):
                torch.manual_seed(1000 * seed + r)
                mm = make()
                train(mm, lambda: loss_on(mm, T), val_loss_fn=lambda: loss_on(mm, V), steps=steps, seed=1000 * seed + r)
                ms.append(mm)
            return lambda X: np.mean([mm.predict(torch.tensor(X)).numpy() for mm in ms], 0)

        net = ens(lambda: BinaryClassifier(X_all.shape[1]), lambda mm, S: mm.loss(S(X_all), S(y_any)))
        glm = ens(lambda: BinaryClassifier(X_all.shape[1], depth=0), lambda mm, S: mm.loss(S(X_all), S(y_any)))
        ours = ens(lambda: OccupancyNet(nt, 0, K + 1, fixed_p=[None] * K + [0.99], p_head=MonotoneDetection(signs, K)),
                   lambda mm, S: mm.loss(S(Xt), S(Xo), S(Y), S(M)))
        orc = ens(lambda: BinaryClassifier(nt), lambda mm, S: mm.loss(S(Xt), S(D["z"])))
        preds = {"Standard (terrain + access)": net(X_all[te]), "Steger-MLP": net(X_neu), "Steger-GLM": glm(X_neu),
                 "OURS": ours(Xt[te]), "Oracle": orc(Xt[te])}
        res[seed] = {k: metrics(v, D) for k, v in preds.items()}
        print(f"  draw {seed}: {time.time() - t0:.0f}s " + " | ".join(
            f"{k} {v['auc_reference']:.3f}/{v['remote_near_ratio']:.2f}" for k, v in res[seed].items()), flush=True)
        out = HERE / "results_q4"; out.mkdir(exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:4], sys.argv[4] if len(sys.argv) > 4 else None)
