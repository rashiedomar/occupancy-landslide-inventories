"""
Q1: does the occupancy model work with GENUINELY biased, real-world landslide records?

Observers = three real, independently produced Gorkha inventories (USGS Ground Failure Database v4):
Gnyawali (2016), Valagussa (2021), Zhang (2016), plus a small exhaustively mapped anchor area taken
from the reference inventory (Roback 2017, 2 blocks ~ 2%). No simulated records anywhere.

Real biases found (q1_real_inventories.py): detection depends on SLOPE (gentle-slope landslides missed),
not on roads; ~40% of each inventory's cells are not in the reference (false positives / disagreement).
So the occupancy model gets (i) terrain-visibility detection features with free shape and (ii) learned
false-positive rates.

Scored against two truths: the reference (Roback) and a consensus truth (cell in >=2 of 4 inventories).
Usage: python q1_run_real.py [--seeds 5]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "toy_simulation"))
sys.path.insert(0, str(HERE))
from models import BinaryClassifier, MonotoneDetection, OccupancyNet, train  # noqa: E402
from run_iburi import _z, block_split, load_grid  # noqa: E402
from run_iburi_v3 import pick_anchor  # noqa: E402

GRID = HERE.parent / "data/processed/gorkha_grid.npz"
OBS = HERE.parent / "data/processed/gorkha_real_obs.npz"
OUT = HERE / "results_q1"; OUT.mkdir(exist_ok=True)
ACC = ["log_d_major", "log_d_public", "log_d_track", "built"]
ACC_SIGNS = [-1, -1, -1, +1]
VIS = ["slope_mean", "slope_max", "relief", "aspect_cos", "aspect_sin"]   # image visibility proxies


def setup(seed, grid, obs):
    d, m, terrain, dist = grid
    rng = np.random.default_rng(seed)
    z = d["z"].astype(np.int8)
    split = block_split(m, rng)
    anchor = pick_anchor(split, m, np.random.default_rng(10_000 + seed))
    Xt = np.stack([_z(terrain[k], m)[m] for k in sorted(terrain)], -1).astype(np.float32)
    acc = {"log_d_major": np.log1p(dist["major"]), "log_d_public": np.log1p(dist["public"]),
           "log_d_track": np.log1p(dist["track"]), "built": d["built_density"]}
    Xa = np.stack([_z(acc[k], m)[m] for k in ACC], -1).astype(np.float32)
    Xv = np.stack([_z(terrain[k], m)[m] for k in VIS], -1).astype(np.float32)
    yr, cr = obs["y"], obs["cover"]
    Y = np.concatenate([yr[:, m].T, ((z == 1) & anchor)[m][:, None]], 1).astype(np.int8)
    M = np.concatenate([cr[:, m].T, anchor[m][:, None]], 1)
    consensus = ((z.astype(int) + yr.sum(0)) >= 2)[m].astype(np.int8)
    sp = split[m]
    return dict(Xt=Xt, Xa=Xa, Xv=Xv, Y=Y, M=M, z=z[m], cons=consensus, slope=terrain["slope_mean"][m],
                d_pub=dist["public"][m], tr=sp == 0, va=sp == 1, te=sp == 2, anchor=anchor[m])


def metrics(p, D):
    te = D["te"]; z, c, sl, dp = D["z"][te], D["cons"][te], D["slope"][te], D["d_pub"][te]
    lo, hi = sl <= np.quantile(sl, 0.25), sl >= np.quantile(sl, 0.75)
    near, far = dp <= np.quantile(dp, 0.25), dp >= np.quantile(dp, 0.75)
    ratio = lambda a, b, t: float((p[a].mean() / p[b].mean()) / (t[a].mean() / t[b].mean()))
    return dict(auc_reference=roc_auc_score(z, p), auc_consensus=roc_auc_score(c, p),
                auc_gentle_quarter=roc_auc_score(z[lo], p[lo]),
                rate_over_reference=float(p.mean() / z.mean()),
                abs_log_rate=float(abs(np.log(p.mean() / z.mean()))),
                steep_gentle_ratio=ratio(hi, lo, z), remote_near_ratio=ratio(far, near, z),
                spearman_vs_oracle=None)


def run_one(seed, grid, obs, restarts=3, steps=3000):
    D = setup(seed, grid, obs)
    Xt, Xa, Xv, Y, M, tr, va, te = D["Xt"], D["Xa"], D["Xv"], D["Y"], D["M"], D["tr"], D["va"], D["te"]
    T = lambda a: torch.tensor(a[tr]); V = lambda a: torch.tensor(a[va])
    y_any = Y.max(1)
    nt = Xt.shape[1]
    X_all = np.concatenate([Xt, Xa], 1)
    X_neu = X_all[te].copy(); X_neu[:, nt:] = 0.0
    Xa_vis = np.concatenate([Xa, Xv, Xv], 1)                       # visibility features with free shape
    signs_vis = ACC_SIGNS + [+1] * len(VIS) + [-1] * len(VIS)
    fixed = [None, None, None, 0.99]

    def ens(make, loss_on):
        ms = []
        for r in range(restarts):
            torch.manual_seed(1000 * seed + r)
            mm = make()
            train(mm, lambda: loss_on(mm, T), val_loss_fn=lambda: loss_on(mm, V), steps=steps, seed=1000 * seed + r)
            ms.append(mm)
        return ms

    pred = lambda ms, X: np.mean([mm.predict(torch.tensor(X)).numpy() for mm in ms], 0)
    preds, learned = {}, {}
    net = ens(lambda: BinaryClassifier(X_all.shape[1]), lambda mm, S: mm.loss(S(X_all), S(y_any)))
    glm = ens(lambda: BinaryClassifier(X_all.shape[1], depth=0), lambda mm, S: mm.loss(S(X_all), S(y_any)))
    ter = ens(lambda: BinaryClassifier(nt), lambda mm, S: mm.loss(S(Xt), S(y_any)))
    preds["Standard (terrain + access)"] = pred(net, X_all[te])
    preds["Standard (terrain only)"] = pred(ter, Xt[te])
    preds["Steger-MLP (access = mean)"] = pred(net, X_neu)
    preds["Steger-GLM (access neutralised)"] = pred(glm, X_neu)

    def occ(name, Xo, signs, fp):
        ms = ens(lambda: OccupancyNet(nt, 0, 4, fixed_p=fixed, p_head=MonotoneDetection(signs, 3),
                                      false_pos=[True, True, True, False] if fp else None),
                 lambda mm, S: mm.loss(S(Xt), S(Xo), S(Y), S(M)))
        preds[name] = pred(ms, Xt[te])
        with torch.no_grad():
            ph = np.mean([mm.predict_detection(torch.tensor(Xo)).numpy() for mm in ms], 0)
            learned[name] = {"mean_detection": [float(v) for v in ph.mean(0)]}
            if fp:
                learned[name]["false_pos_rate"] = [float(v) for v in
                                                   np.mean([torch.sigmoid(mm.fp_logit).numpy() for mm in ms], 0)[:3]]

    occ("OURS (access detection)", Xa, ACC_SIGNS, fp=False)
    occ("OURS + visibility detection", Xa_vis, signs_vis, fp=False)
    occ("OURS + visibility + false positives", Xa_vis, signs_vis, fp=True)
    orc = ens(lambda: BinaryClassifier(nt), lambda mm, S: mm.loss(S(Xt), S(D["z"])))
    preds["Oracle (reference inventory)"] = pred(orc, Xt[te])

    out = {k: metrics(v, D) for k, v in preds.items()}
    for k in out:
        out[k]["spearman_vs_oracle"] = float(spearmanr(preds[k], preds["Oracle (reference inventory)"]).statistic)
    out["_learned"] = learned
    out["_info"] = dict(n_test=int(te.sum()), true_rate=float(D["z"].mean()),
                        records_per_inventory=[int(v) for v in Y.sum(0)],
                        union_recall=float(Y[:, :3].max(1)[D["z"] == 1].mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    grid = load_grid(GRID)
    obs = dict(np.load(OBS))
    res = {}
    for s in range(a.seeds):
        t0 = time.time()
        res[s] = run_one(s, grid, obs)
        print(f"draw {s}: {time.time() - t0:.0f}s {res[s]['_info']}", flush=True)
        (OUT / "raw_results.json").write_text(json.dumps(res, indent=1, default=float))
    print("finished", flush=True)


if __name__ == "__main__":
    main()
