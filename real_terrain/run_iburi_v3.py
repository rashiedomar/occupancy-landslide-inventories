"""
Stage 3: stabilising the occupancy model on real Iburi terrain.

Same data, same simulated biased inventories and same 3 km block split as Stage 2 (run_iburi.py,
identical random draws), plus three changes tested one at a time:

  (1) Ensembling   : every model = mean of 3 independently trained restarts (was best-of-3).
  (2) Monotone     : detection head is an additive, sign-constrained model (MonotoneDetection):
                     detection can only stay flat or drop with distance to roads and with cloud,
                     and only stay flat or rise with built-up density.
  Scenario C (stress test): detection is MISSPECIFIED for the model -- it also depends on a hidden
                     "reporting effort" field the model never sees, on a road x built-up interaction,
                     and agency surveys skip steep slopes (detection depends on terrain).
  (3) Anchor area  : 2 random 3 km x 3 km training blocks (~2% of the area) were mapped
                     exhaustively (detection = 1). The model is told this (fixed p = 0.99).
                     Competitor: a classifier trained ONLY on the anchor area.

Usage: python run_iburi_v3.py [--seeds 5] [--restarts 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "toy_simulation"))
sys.path.insert(0, str(HERE))
from models import BinaryClassifier, MonotoneDetection, OccupancyNet, train  # noqa: E402
from run_iburi import _z, block_split, load_grid, metrics, simulate_inventories  # noqa: E402

torch.set_num_threads(2)
OUT = HERE / "results_v3"; OUT.mkdir(exist_ok=True)

ACCESS = ["log_d_major", "log_d_public", "log_d_track", "built", "cloud"]
SIGNS = [-1, -1, -1, +1, -1]          # known direction of each feature's effect on detection
BLOCK = 25                            # 25 cells x 120 m = 3 km


def simulate_misspecified(z, m, terrain, dist, built, rng):
    """Like simulate_inventories, but detection breaks the model's assumptions."""
    from scipy import ndimage
    from run_iburi import _sig
    H, W = z.shape
    cloud = ndimage.gaussian_filter(rng.standard_normal((H, W)), 25)
    cloud = 0.6 * _z(cloud, m) + 0.4 * _z(terrain["elev"], m)
    hidden = _z(ndimage.gaussian_filter(rng.standard_normal((H, W)), 15), m)   # unobserved effort
    bz = _z(built, m)
    steep = (_z(terrain["slope_mean"], m) > 1.0).astype(float)
    p = np.stack([
        _sig(-1.0 - 2.0 * dist["major"] + 0.8 * bz + 0.8 * hidden - 0.8 * dist["major"] * bz),
        _sig(0.5 - 5.0 * dist["public"] + 0.8 * hidden - 1.5 * steep),
        _sig(0.0 - 1.8 * _z(cloud, m) - 0.8 * steep * (terrain["aspect_cos"] > 0.5)),  # shadowed N slopes
    ])
    cover = np.ones_like(p, bool)
    sat = np.zeros((H, W), bool)
    for _ in range(3):
        h, w = rng.integers(45, 90), rng.integers(40, 80)
        r0, c0 = rng.integers(0, H - h), rng.integers(0, W - w)
        sat[r0:r0 + h, c0:c0 + w] = True
    cover[2] = sat
    y = ((rng.random(p.shape) < p) & cover & (z[None] == 1) & m[None]).astype(np.int8)
    return y, p, cover, cloud


def pick_anchor(split, m, rng, n_blocks=2):
    """Choose n random training blocks that lie mostly inside the mapped area."""
    H, W = m.shape
    cands = []
    for r in range(0, H, BLOCK):
        for c in range(0, W, BLOCK):
            sl = (slice(r, r + BLOCK), slice(c, c + BLOCK))
            if (split[sl] == 0).all() and m[sl].mean() > 0.9:
                cands.append(sl)
    anchor = np.zeros((H, W), bool)
    for i in rng.choice(len(cands), n_blocks, replace=False):
        anchor[cands[i]] = True
    return anchor


def run_one(seed, scen, steps, restarts, grid):
    d, m, terrain, dist = grid
    rng = np.random.default_rng(seed)
    z = d["z"].astype(np.int8)
    built = d["built_density"]
    sim = simulate_misspecified if scen == "C" else simulate_inventories
    y, p_true, cover, cloud = sim(z, m, terrain, dist, built, rng)                    # A/B: same as Stage 2
    split = block_split(m, rng)                                                       # same as Stage 2
    anchor = pick_anchor(split, m, np.random.default_rng(10_000 + seed))
    y_anchor = (z == 1) & anchor

    T_NAMES = sorted(terrain)
    Xt = np.stack([_z(terrain[k], m)[m] for k in T_NAMES], -1).astype(np.float32)
    acc = {"log_d_major": np.log1p(dist["major"]), "log_d_public": np.log1p(dist["public"]),
           "log_d_track": np.log1p(dist["track"]), "built": built, "cloud": cloud}
    Xa = np.stack([_z(acc[k], m)[m] for k in ACCESS], -1).astype(np.float32)
    X_site = np.concatenate([Xt, Xa[:, :3]], 1) if scen == "B" else Xt
    Y3, M3 = y[:, m].T, cover[:, m].T
    Y4 = np.concatenate([Y3, y_anchor[m][:, None]], 1).astype(np.int8)
    M4 = np.concatenate([M3, anchor[m][:, None]], 1)
    zz = z[m]
    y_any = Y4.max(1)                         # standard practice merges every record, incl. anchor
    sp = split[m]; tr, va, te = sp == 0, sp == 1, sp == 2
    anc = anchor[m]

    def ensemble(make, loss_on, tr_mask, va_mask, predict):
        Ttr = lambda a: torch.tensor(a[tr_mask]); Tva = lambda a: torch.tensor(a[va_mask])
        outs, models = [], []
        for r in range(restarts):
            torch.manual_seed(1000 * seed + r)
            mdl = make()
            train(mdl, lambda: loss_on(mdl, Ttr), val_loss_fn=lambda: loss_on(mdl, Tva),
                  steps=steps, seed=1000 * seed + r)
            outs.append(predict(mdl)); models.append(mdl)
        return np.mean(outs, 0), models

    preds, keep = {}, {}

    def clf(X, label, name, tr_mask=tr, va_mask=va):
        preds[name], _ = ensemble(lambda: BinaryClassifier(X.shape[1]),
                                  lambda mm, S: mm.loss(S(X), S(label)), tr_mask, va_mask,
                                  lambda mm: mm.predict(torch.tensor(X[te])).numpy())

    def occ(name, monotone, use_anchor):
        Y, M = (Y4, M4) if use_anchor else (Y3, M3)
        K = Y.shape[1]
        fixed = [None, None, None, 0.99] if use_anchor else None
        make = (lambda: OccupancyNet(X_site.shape[1], Xa.shape[1], K, fixed_p=fixed,
                                     p_head=MonotoneDetection(SIGNS, 3))) if monotone else \
               (lambda: OccupancyNet(X_site.shape[1], Xa.shape[1], K, fixed_p=fixed))
        preds[name], keep[name] = ensemble(make, lambda mm, S: mm.loss(S(X_site), S(Xa), S(Y), S(M)),
                                           tr, va, lambda mm: mm.predict(torch.tensor(X_site[te])).numpy())

    clf(np.concatenate([Xt, Xa], 1), y_any, "Standard (terrain + access)")
    clf(Xt, y_any, "Standard (terrain only)")
    clf(X_site, zz, "Oracle (full real inventory)")
    # anchor-only: train on exhaustively mapped blocks; early-stop on a random 20% of them
    r_an = np.random.default_rng(seed).random(len(zz)) < 0.2
    clf(Xt, zz, "Anchor area only (terrain)", tr_mask=anc & ~r_an, va_mask=anc & r_an)
    occ("Occupancy v1 (MLP detection)", monotone=False, use_anchor=False)
    occ("Occupancy + monotone", monotone=True, use_anchor=False)
    occ("Occupancy + anchor", monotone=False, use_anchor=True)
    occ("Occupancy + monotone + anchor (FULL)", monotone=True, use_anchor=True)

    d_pub = dist["public"][m][te]
    out = {k: metrics(v, zz[te], d_pub) for k, v in preds.items()}
    for k in out:
        out[k]["abs_log_ratio"] = float(abs(np.log(out[k]["remote_near_ratio"])))
    # detection recovery of the full model (ensemble mean of learned detection probs)
    full = keep["Occupancy + monotone + anchor (FULL)"]
    with torch.no_grad():
        p_hat = np.mean([mm.predict_detection(torch.tensor(Xa)).numpy() for mm in full], 0)
    pt = p_true[:, m].T
    out["_detection_recovery"] = {n: float(np.corrcoef(p_hat[M3[:, k], k], pt[M3[:, k], k])[0, 1])
                                  for k, n in enumerate(["news", "agency", "satellite"])}
    out["_info"] = dict(anchor_cells=int(anc.sum()), anchor_landslides=int(zz[anc].sum()),
                        frac_recorded=float(y_any[zz == 1].mean()))
    extras = dict(p_hat=p_hat, p_true=pt, cover=M3, Xa=Xa,
                  curves=[mm.p_head for mm in full])
    return out, preds, te, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--scenarios", default="AB")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    grid = load_grid()
    res_path = OUT / f"raw_results_{args.scenarios}.json"
    res = {}
    for scen in args.scenarios:
        res[scen] = []
        for s in range(args.seeds):
            t0 = time.time()
            out, preds, te, ex = run_one(s, scen, args.steps, args.restarts, grid)
            res[scen].append(out)
            print(f"[scenario {scen}] seed {s}: {time.time() - t0:.0f}s  anchor: "
                  f"{out['_info']['anchor_cells']} cells, {out['_info']['anchor_landslides']} landslide cells",
                  flush=True)
            if s == 0:
                np.savez_compressed(OUT / f"seed0_{scen}.npz", test=te, p_hat=ex["p_hat"],
                                    p_true=ex["p_true"], cover=ex["cover"], Xa=ex["Xa"], **preds)
                torch.save([h.state_dict() for h in ex["curves"]], OUT / f"seed0_{scen}_detection_heads.pt")
            res_path.write_text(json.dumps(res, indent=1, default=float))
    summarize()


def summarize():
    res = {}
    for f in sorted(OUT.glob("raw_results_*.json")):
        res.update(json.loads(f.read_text()))
    titles = {"A": "A: susceptibility sees terrain only", "B": "B: susceptibility also sees road distance",
              "C": "C: stress test - detection misspecified (hidden effort, interactions, steep-slope gaps)"}
    lines = []
    for scen, runs in res.items():
        lines.append(f"\n=== Scenario {titles[scen]} ({len(runs)} draws; mean ± sd; test = 3 km blocks; "
                     "scored vs FULL real inventory) ===")
        models = [k for k in runs[0] if not k.startswith("_")]
        mets = ["auc_true_landslides", "auc_remote_quarter", "mean_pred_over_true",
                "remote_near_ratio", "abs_log_ratio"]
        lines.append(f"{'model':40s}" + "".join(f"{x:>22s}" for x in mets) + f"{'ratio median':>14s}")
        for mdl in models:
            row = f"{mdl:40s}"
            for met in mets:
                v = np.array([r[mdl][met] for r in runs])
                row += f"{v.mean():>14.3f} ± {v.std():.3f}"
            row += f"{np.median([r[mdl]['remote_near_ratio'] for r in runs]):>14.2f}"
            lines.append(row)
        det = {n: np.mean([r["_detection_recovery"][n] for r in runs]) for n in runs[0]["_detection_recovery"]}
        lines.append("FULL model detection-head corr. with truth: " +
                     ", ".join(f"{n}={v:.2f}" for n, v in det.items()))
    txt = "\n".join(lines)
    print(txt)
    (OUT / "summary.txt").write_text(txt)


if __name__ == "__main__":
    main()
