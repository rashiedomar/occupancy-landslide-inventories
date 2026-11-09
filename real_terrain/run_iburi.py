"""
Semi-synthetic experiment on REAL terrain, REAL roads and REAL landslides.

Truth  z  = the near-complete 2018 Iburi earthquake landslide inventory (5,625 polygons).
We then pretend we never had it: we simulate three biased "observers" that each record only
some of those real landslides, with KNOWN detection functions:
   1. news catalog      : decays with distance to major roads, higher near built-up areas
   2. agency field survey: decays with distance to public roads
   3. satellite event map: road-independent, blocked by (synthetic) cloud, partial footprints
Models only see these biased records. They are scored against the full real inventory.

Scenario A: susceptibility head sees terrain only.
Scenario B: susceptibility head ALSO sees road distances (common practice; lets the model blame
            roads, which is exactly what reporting bias tempts it to do).

Usage: python run_iburi.py [--seeds 5] [--restarts 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "toy_simulation"))
from models import BinaryClassifier, OccupancyNet, train  # noqa: E402

torch.set_num_threads(2)
DATA = HERE.parent / "data/processed/iburi_grid.npz"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)

EPICENTER_LATLON = (42.686, 141.929)   # JMA, 2018-09-06


def _z(a, m):
    return (a - a[m].mean()) / (a[m].std() + 1e-9)


def _sig(x):
    return 1 / (1 + np.exp(-x))


def load_grid(path=None):
    d = np.load(path or DATA, allow_pickle=True)
    m = d["mask"]
    terrain = {k.split("__")[1]: d[k] for k in d.files if k.startswith("terrain__")}
    dist = {k.split("__")[1]: d[k] for k in d.files if k.startswith("dist__")}
    if "epi_dist_km" in terrain:              # grids from prep_generic.py store their own epicentres
        return d, m, terrain, dist
    # epicentral distance (km) on the grid (Iburi grid from prep_iburi.py)
    from pyproj import Transformer
    ex, ey = Transformer.from_crs("EPSG:4326", str(d["crs"]), always_xy=True).transform(
        EPICENTER_LATLON[1], EPICENTER_LATLON[0])
    H, W = m.shape
    cell, x0, y1 = float(d["cell"]), float(d["x0"]), float(d["y1"])
    xs = x0 + (np.arange(W) + 0.5) * cell
    ys = y1 - (np.arange(H) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    terrain["epi_dist_km"] = np.hypot(X - ex, Y - ey) / 1000.0
    return d, m, terrain, dist


def simulate_inventories(z, m, terrain, dist, built, rng):
    """Thin the real inventory with known detection functions. Returns y, p, cover (K,H,W)."""
    H, W = z.shape
    cloud = ndimage.gaussian_filter(rng.standard_normal((H, W)), 25)
    cloud = 0.6 * _z(cloud, m) + 0.4 * _z(terrain["elev"], m)
    p = np.stack([
        _sig(-1.0 - 2.0 * dist["major"] + 0.8 * _z(built, m)),     # news catalog
        _sig(0.5 - 5.0 * dist["public"]),                          # agency field survey
        _sig(0.0 - 1.8 * _z(cloud, m)),                            # satellite event map
    ])
    cover = np.ones_like(p, bool)
    sat = np.zeros((H, W), bool)                                   # 3 random event footprints
    for _ in range(3):
        h, w = rng.integers(45, 90), rng.integers(40, 80)
        r0, c0 = rng.integers(0, H - h), rng.integers(0, W - w)
        sat[r0:r0 + h, c0:c0 + w] = True
    cover[2] = sat
    y = ((rng.random(p.shape) < p) & cover & (z[None] == 1) & m[None]).astype(np.int8)
    return y, p, cover, cloud


def block_split(m, rng, block=25, frac=(0.6, 0.15, 0.25)):
    """Random 3 km x 3 km blocks -> train / val / test (0/1/2)."""
    H, W = m.shape
    nb = (int(np.ceil(H / block)), int(np.ceil(W / block)))
    lab = rng.choice(3, size=nb, p=frac)
    lab = np.kron(lab, np.ones((block, block), int))[:H, :W]
    return lab


def metrics(pred, z, d_pub):
    near = d_pub <= np.quantile(d_pub, 0.25)
    remote = d_pub >= np.quantile(d_pub, 0.75)
    return dict(
        auc_true_landslides=roc_auc_score(z, pred),
        auc_remote_quarter=roc_auc_score(z[remote], pred[remote]),
        mean_pred_over_true=float(pred.mean() / z.mean()),
        remote_near_ratio=float((pred[remote].mean() / pred[near].mean()) /
                                (z[remote].mean() / z[near].mean())),
    )


def run_one(seed, scen, steps, restarts, grid):
    d, m, terrain, dist = grid
    rng = np.random.default_rng(seed)
    z = d["z"].astype(np.int8)
    built = d["built_density"]
    y, p_true, cover, cloud = simulate_inventories(z, m, terrain, dist, built, rng)
    split = block_split(m, rng)

    T_NAMES = sorted(terrain)
    Xt = np.stack([_z(terrain[k], m)[m] for k in T_NAMES], -1).astype(np.float32)
    acc = {"log_d_major": np.log1p(dist["major"]), "log_d_public": np.log1p(dist["public"]),
           "log_d_track": np.log1p(dist["track"]), "built": built,
           "cloud": cloud}   # cloud cover is observable from the imagery archive -> detection feature
    Xa = np.stack([_z(v, m)[m] for v in acc.values()], -1).astype(np.float32)
    X_site = np.concatenate([Xt, Xa[:, :3]], 1) if scen == "B" else Xt
    Y = y[:, m].T; M = cover[:, m].T
    zz = z[m]; y_any = Y.max(1)
    sp = split[m]
    tr, va, te = sp == 0, sp == 1, sp == 2

    T = lambda a: torch.tensor(a[tr]); V = lambda a: torch.tensor(a[va])
    preds = {}

    def best_of(make, loss_on):
        best, best_m = float("inf"), None
        for r in range(restarts):
            torch.manual_seed(1000 * seed + r)
            mdl = make()
            train(mdl, lambda: loss_on(mdl, T), val_loss_fn=lambda: loss_on(mdl, V),
                  steps=steps, seed=1000 * seed + r)
            with torch.no_grad():
                v = loss_on(mdl, V).item()
            if v < best:
                best, best_m = v, mdl
        return best_m

    def fit_clf(X, label, name):
        mdl = best_of(lambda: BinaryClassifier(X.shape[1]), lambda mm, S: mm.loss(S(X), S(label)))
        preds[name] = mdl.predict(torch.tensor(X[te])).numpy()

    def fit_occ(ks, name, merge=False):
        Yk, Mk = Y[:, ks], M[:, ks]
        if merge:
            Yk = Yk.max(1, keepdims=True); Mk = np.ones_like(Yk, bool)
        mdl = best_of(lambda: OccupancyNet(X_site.shape[1], Xa.shape[1], Yk.shape[1]),
                      lambda mm, S: mm.loss(S(X_site), S(Xa), S(Yk), S(Mk)))
        preds[name] = mdl.predict(torch.tensor(X_site[te])).numpy()
        return mdl

    fit_clf(np.concatenate([Xt, Xa], 1), y_any, "Naive (terrain + access)")
    fit_clf(Xt, y_any, "Naive (terrain only)")
    fit_clf(X_site, zz, "Oracle (trained on full real inventory)")
    fit_occ([0, 1, 2], "Occupancy, 1 merged inventory", merge=True)
    fit_occ([0, 1], "Occupancy, 2 road-biased inventories")
    full = fit_occ([0, 1, 2], "Occupancy, 3 inventories (OURS)")

    d_pub = dist["public"][m][te]
    out = {k: metrics(v, zz[te], d_pub) for k, v in preds.items()}
    p_hat = full.predict_detection(torch.tensor(Xa)).numpy()
    pt = p_true[:, m].T
    out["_detection_recovery"] = {n: float(np.corrcoef(p_hat[M[:, k], k], pt[M[:, k], k])[0, 1])
                                  for k, n in enumerate(["news", "agency", "satellite"])}
    out["_info"] = dict(frac_recorded=float(y_any[zz == 1].mean()),
                        n_recorded=int(y_any.sum()), n_true=int(zz.sum()),
                        per_inventory=[int(v) for v in Y.sum(0)], n_test=int(te.sum()))
    return out, preds, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=3)
    args = ap.parse_args()
    grid = load_grid()
    res = {}
    for scen in ["A", "B"]:
        res[scen] = []
        for s in range(args.seeds):
            t0 = time.time()
            out, preds, te = run_one(s, scen, args.steps, args.restarts, grid)
            res[scen].append(out)
            print(f"[scenario {scen}] seed {s}: {time.time() - t0:.0f}s, "
                  f"recorded {out['_info']['frac_recorded']:.0%} of real landslides "
                  f"{out['_info']['per_inventory']}", flush=True)
            if s == 0:
                np.savez_compressed(OUT / f"seed0_{scen}.npz", test=te, **preds)
            (OUT / "raw_results.json").write_text(json.dumps(res, indent=1, default=float))

    lines = []
    names = {"A": "A: susceptibility sees terrain only", "B": "B: susceptibility also sees road distance"}
    for scen, runs in res.items():
        lines.append(f"\n=== Scenario {names[scen]}  ({len(runs)} seeds; test = random 3 km blocks; "
                     f"scored vs FULL real inventory) ===")
        models = [k for k in runs[0] if not k.startswith("_")]
        mets = list(runs[0][models[0]])
        lines.append(f"{'model':42s}" + "".join(f"{x:>24s}" for x in mets))
        for mdl in models:
            row = f"{mdl:42s}"
            for met in mets:
                v = np.array([r[mdl][met] for r in runs])
                row += f"{v.mean():>16.3f} ± {v.std():.3f}"
            lines.append(row)
        det = {n: np.mean([r["_detection_recovery"][n] for r in runs]) for n in runs[0]["_detection_recovery"]}
        rec = np.mean([r["_info"]["frac_recorded"] for r in runs])
        lines.append(f"landslides recorded by any inventory: {rec:.0%}.  detection-head corr. with truth: " +
                     ", ".join(f"{n}={v:.2f}" for n, v in det.items()))
    txt = "\n".join(lines)
    print(txt)
    (OUT / "summary.txt").write_text(txt)


if __name__ == "__main__":
    main()
