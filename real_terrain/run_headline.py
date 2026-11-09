"""
Headline comparison on ANY prepared grid (Iburi or Gorkha): the models that carry the claims.

  Standard (terrain + access)          : common practice
  Steger-MLP (access = mean)           : strongest established fix (fit with bias variables, neutralise)
  Steger-GLM (access neutralised)      : Steger's original logistic-regression setting
  OURS full                            : occupancy model, monotone detection + anchor, terrain-only susceptibility
  OURS + short-range road              : same, plus <=300 m non-negative road-cut term
  Oracle                               : trained on the full inventory (upper bound)

Scenario A = reporting bias as in Stage 2; C = misspecified detection stress test.
Usage: python run_headline.py --grid ../data/processed/gorkha_grid.npz --out results_gorkha --scenario A
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
from models import BinaryClassifier, MonotoneDetection, OccupancyNet, PsiWithRoadCut, train  # noqa: E402
from run_iburi import _z, block_split, load_grid, metrics, simulate_inventories  # noqa: E402
from run_iburi_v3 import ACCESS, SIGNS, pick_anchor, simulate_misspecified  # noqa: E402


def build(seed, scen, grid):
    d, m, terrain, dist = grid
    rng = np.random.default_rng(seed)
    z = d["z"].astype(np.int8)
    built = d["built_density"]
    sim = simulate_misspecified if scen == "C" else simulate_inventories
    y, _, cover, cloud = sim(z, m, terrain, dist, built, rng)
    split = block_split(m, rng)
    anchor = pick_anchor(split, m, np.random.default_rng(10_000 + seed))
    Xt = np.stack([_z(terrain[k], m)[m] for k in sorted(terrain)], -1).astype(np.float32)
    acc = {"log_d_major": np.log1p(dist["major"]), "log_d_public": np.log1p(dist["public"]),
           "log_d_track": np.log1p(dist["track"]), "built": built, "cloud": cloud}
    Xa = np.stack([_z(acc[k], m)[m] for k in ACCESS], -1).astype(np.float32)
    Y = np.concatenate([y[:, m].T, ((z == 1) & anchor)[m][:, None]], 1).astype(np.int8)
    M = np.concatenate([cover[:, m].T, anchor[m][:, None]], 1)
    sp = split[m]
    info = dict(n_cells=int(m.sum()), n_true=int(z[m].sum()), n_test=int((sp == 2).sum()),
                frac_recorded=float(Y[:, :3].max(1)[z[m] == 1].mean()), per_inventory=[int(v) for v in Y.sum(0)])
    return dict(Xt=Xt, Xa=Xa, Y=Y, M=M, z=z[m], d_pub=dist["public"][m].astype(np.float32),
                tr=sp == 0, va=sp == 1, te=sp == 2), info


def run_one(seed, scen, grid, restarts=3, steps=3000, skip_roadcut=False):
    D, info = build(seed, scen, grid)
    Xt, Xa, Y, M, tr, va, te = D["Xt"], D["Xa"], D["Y"], D["M"], D["tr"], D["va"], D["te"]
    T = lambda a: torch.tensor(a[tr]); V = lambda a: torch.tensor(a[va])
    y_any = Y.max(1)
    X_all = np.concatenate([Xt, Xa], 1); nt = Xt.shape[1]
    X_neutral = X_all[te].copy(); X_neutral[:, nt:] = 0.0
    X_cut = np.concatenate([Xt, D["d_pub"][:, None]], 1)
    fixed = [None, None, None, 0.99]

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
    full = ens(lambda: OccupancyNet(nt, 0, 4, fixed_p=fixed, p_head=MonotoneDetection(SIGNS, 3)),
               lambda mm, S: mm.loss(S(Xt), S(Xa), S(Y), S(M)))
    cut = None if skip_roadcut else ens(
        lambda: OccupancyNet(nt + 1, 0, 4, fixed_p=fixed, p_head=MonotoneDetection(SIGNS, 3), psi_head=PsiWithRoadCut(nt)),
        lambda mm, S: mm.loss(S(X_cut), S(Xa), S(Y), S(M)))
    oracle = ens(lambda: BinaryClassifier(nt), lambda mm, S: mm.loss(S(Xt), S(D["z"])))
    preds = {"Standard (terrain + access)": net(X_all[te]),
             "Steger-MLP (access = mean)": net(X_neutral),
             "Steger-GLM (access neutralised)": glm(X_neutral),
             "OURS full": full(Xt[te]),
             "Oracle": oracle(Xt[te])}
    if cut is not None:
        preds["OURS + short-range road"] = cut(X_cut[te])
    out = {}
    for k, p in preds.items():
        out[k] = metrics(p, D["z"][te], D["d_pub"][te])
        out[k]["abs_log_ratio"] = float(abs(np.log(out[k]["remote_near_ratio"])))
    out["_info"] = info
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", default="A")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-list", default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--skip-roadcut", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out_dir = HERE / a.out; out_dir.mkdir(exist_ok=True)
    grid = load_grid(a.grid)
    seeds = [int(v) for v in a.seed_list.split(",")] if a.seed_list else list(range(a.seeds))
    tag = a.scenario + ("_" + "".join(map(str, seeds)) if a.seed_list else "")
    res = {}
    for s in seeds:
        t0 = time.time()
        res[s] = run_one(s, a.scenario, grid, skip_roadcut=a.skip_roadcut)
        print(f"[{a.scenario}] draw {s}: {time.time() - t0:.0f}s {res[s]['_info']}", flush=True)
        (out_dir / f"raw_results_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print("finished", flush=True)


if __name__ == "__main__":
    main()
