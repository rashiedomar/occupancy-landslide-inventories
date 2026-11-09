"""
Run the toy experiment: does the occupancy model recover TRUE landslide susceptibility
from road-biased inventories better than standard practice?

Usage:  python run_experiments.py            (5 seeds, both scenarios, ~ a few min on CPU)
        python run_experiments.py --seeds 1  (quick check)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from models import BinaryClassifier, OccupancyNet, train
from simulate import make_world

torch.set_num_threads(2)
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

TERRAIN = ["slope", "curvature", "weak_rock", "rain", "ndvi", "elev"]
ACCESS = ["dist_road", "pop", "cloud"]


def stack(world, names, src):
    return np.stack([src[n].ravel() for n in names], -1).astype(np.float32)


def metrics(pred, world, test):
    psi, z = world.psi.ravel()[test], world.z.ravel()[test]
    d = world.dist_road_km.ravel()[test]
    near = d <= np.quantile(d, 0.25)          # nearest quarter of the test area to a road
    remote = d >= np.quantile(d, 0.75)        # most remote quarter
    return dict(
        auc_true_landslides=roc_auc_score(z, pred),
        auc_remote_quarter=roc_auc_score(z[remote], pred[remote]),
        spearman_vs_true_psi=spearmanr(pred, psi).statistic,
        mean_pred_over_true=float(pred.mean() / psi.mean()),
        mae_vs_true_psi=float(np.abs(pred - psi).mean()),
        # how the model ranks remote vs near-road ground, relative to the truth (1.0 = unbiased,
        # <1 = remote slopes under-rated, which is exactly the inventory-bias failure)
        remote_near_ratio=float((pred[remote].mean() / pred[near].mean()) /
                                (psi[remote].mean() / psi[near].mean())),
    )


def run_one(seed, road_effect, steps, restarts=3):
    w = make_world(seed=seed, road_effect=road_effect)
    H, W = w.shape
    cols = np.tile(np.arange(W), H)
    # spatial block split (west -> east): train 56% | validation 14% | test 30%
    train_idx = cols < int(0.56 * W)
    val_idx = (cols >= int(0.56 * W)) & (cols < int(0.7 * W))
    test_idx = cols >= int(0.7 * W)

    Xt = stack(w, TERRAIN, w.terrain)
    Xa = stack(w, ACCESS, w.access)
    X_site = np.concatenate([Xt, Xa[:, :1]], 1) if road_effect else Xt   # psi may see roads in scen. B
    Y = w.y.reshape(len(w.inventory_names), -1).T                         # (N, K)
    M = w.cover.reshape(len(w.inventory_names), -1).T
    y_any = Y.max(1)
    z = w.z.ravel()

    T = lambda a: torch.tensor(a[train_idx])
    V = lambda a: torch.tensor(a[val_idx])
    out, preds, extra = {}, {}, {}

    def best_of(make, loss_on):
        """Train `restarts` random inits with early stopping; keep best validation loss.
        Validation loss uses the same (observed) labels the model is trained on."""
        best, best_m = float("inf"), None
        for r in range(restarts):
            torch.manual_seed(1000 * seed + r)
            m = make()
            train(m, lambda: loss_on(m, T), val_loss_fn=lambda: loss_on(m, V),
                  steps=steps, seed=1000 * seed + r)
            with torch.no_grad():
                v = loss_on(m, V).item()
            if v < best:
                best, best_m = v, m
        return best_m

    def fit_clf(X, label, name):
        m = best_of(lambda: BinaryClassifier(X.shape[1]), lambda m, S: m.loss(S(X), S(label)))
        preds[name] = m.predict(torch.tensor(X[test_idx])).numpy()

    fit_clf(np.concatenate([Xt, Xa], 1), y_any, "Naive (terrain + access)")
    fit_clf(Xt, y_any, "Naive (terrain only)")
    fit_clf(X_site, z, "Oracle (trained on true landslides)")

    def fit_occ(cols_k, name, merge=False):
        Yk, Mk = Y[:, cols_k], M[:, cols_k]
        if merge:                              # treat union of inventories as ONE observer
            Yk = Yk.max(1, keepdims=True)
            Mk = np.ones_like(Yk, dtype=bool)
        m = best_of(lambda: OccupancyNet(X_site.shape[1], Xa.shape[1], Yk.shape[1]),
                    lambda m, S: m.loss(S(X_site), S(Xa), S(Yk), S(Mk)))
        preds[name] = m.predict(torch.tensor(X_site[test_idx])).numpy()
        extra[name] = m

    fit_occ([0, 1, 2], "Occupancy, 1 merged inventory", merge=True)
    fit_occ([0, 1], "Occupancy, 2 road-biased inventories")
    fit_occ([0, 1, 2], "Occupancy, 3 inventories (OURS)")

    for k, v in preds.items():
        out[k] = metrics(v, w, test_idx)

    # detection-head recovery for the full model (on all cells, where covered)
    m = extra["Occupancy, 3 inventories (OURS)"]
    p_hat = m.predict_detection(torch.tensor(Xa)).numpy()
    p_true = w.p.reshape(3, -1).T
    out["_detection_recovery"] = {
        n: float(np.corrcoef(p_hat[M[:, k], k], p_true[M[:, k], k])[0, 1])
        for k, n in enumerate(w.inventory_names)}
    out["_world"] = w.meta
    return out, w, preds, test_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=3)
    args = ap.parse_args()

    all_res = {}
    for scen, road_effect in [("A_reporting_bias_only", False), ("B_roads_also_cause_landslides", True)]:
        all_res[scen] = []
        for s in range(args.seeds):
            t0 = time.time()
            res, w, preds, test_idx = run_one(s, road_effect, args.steps, args.restarts)
            all_res[scen].append(res)
            print(f"[{scen}] seed {s} done in {time.time() - t0:.1f}s  "
                  f"(landslides recorded: {w.meta['frac_landslides_recorded']:.0%})")
            if s == 0:
                np.savez_compressed(OUT / f"seed0_{scen}.npz",
                                    test_idx=test_idx, psi=w.psi, z=w.z, roads=w.roads,
                                    dist=w.dist_road_km, y_any=w.y.max(0), cover=w.cover,
                                    **{k: v for k, v in preds.items()})
    (OUT / "raw_results.json").write_text(json.dumps(all_res, indent=1, default=float))

    # summary table
    lines = []
    for scen, runs in all_res.items():
        lines.append(f"\n=== {scen}  ({len(runs)} seeds, mean ± sd, test = held-out east 30%, scored vs TRUE landslides) ===")
        models = [k for k in runs[0] if not k.startswith("_")]
        mets = list(runs[0][models[0]].keys())
        lines.append(f"{'model':42s}" + "".join(f"{m:>24s}" for m in mets))
        for mdl in models:
            row = f"{mdl:42s}"
            for met in mets:
                vals = np.array([r[mdl][met] for r in runs])
                row += f"{vals.mean():>16.3f} ± {vals.std():.3f}"
            lines.append(row)
        det = {n: np.mean([r["_detection_recovery"][n] for r in runs]) for n in runs[0]["_detection_recovery"]}
        lines.append("detection-head correlation with TRUE detection prob: " +
                     ", ".join(f"{n}={v:.2f}" for n, v in det.items()))
    txt = "\n".join(lines)
    print(txt)
    (OUT / "summary.txt").write_text(txt)


if __name__ == "__main__":
    main()
