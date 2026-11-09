"""
Stage 4: two fixes for the Stage 3 failure modes, on the SAME draws as Stage 3.

  + spatial      : detection head gets a smooth, penalised spatial field per inventory to absorb
                   unobserved reporting effort (SpatialMonotoneDetection).
  + short-range  : (scenario B) instead of feeding road distance freely into susceptibility, roads may
                   only raise susceptibility through a short-range, non-negative road-cut term with
                   range <= 300 m (PsiWithRoadCut) -- "scale separation" between physical road effects
                   (hundreds of metres) and reporting bias (kilometres).

Only the new variants are trained; baselines and the Stage 3 FULL model are read from
results_v3/ (identical data, anchors and splits for each draw).

Usage: python run_iburi_v4.py --scenario B   (or C)
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
from models import OccupancyNet, PsiWithRoadCut, SpatialMonotoneDetection, rbf_basis, train  # noqa: E402
from run_iburi import _z, block_split, load_grid, metrics, simulate_inventories  # noqa: E402
from run_iburi_v3 import ACCESS, SIGNS, pick_anchor, simulate_misspecified  # noqa: E402

OUT = HERE / "results_v4"; OUT.mkdir(exist_ok=True)
KNOT_SPACING_KM = 3.0


def prepare(seed, scen, grid):
    """Exactly the data of run_iburi_v3.run_one for this seed/scenario."""
    d, m, terrain, dist = grid
    rng = np.random.default_rng(seed)
    z = d["z"].astype(np.int8)
    built = d["built_density"]
    sim = simulate_misspecified if scen == "C" else simulate_inventories
    y, p_true, cover, cloud = sim(z, m, terrain, dist, built, rng)
    split = block_split(m, rng)
    anchor = pick_anchor(split, m, np.random.default_rng(10_000 + seed))
    Xt = np.stack([_z(terrain[k], m)[m] for k in sorted(terrain)], -1).astype(np.float32)
    acc = {"log_d_major": np.log1p(dist["major"]), "log_d_public": np.log1p(dist["public"]),
           "log_d_track": np.log1p(dist["track"]), "built": built, "cloud": cloud}
    Xa = np.stack([_z(acc[k], m)[m] for k in ACCESS], -1).astype(np.float32)
    H, W = m.shape
    cell = float(d["cell"]) / 1000.0
    yy, xx = np.mgrid[0:H, 0:W]
    loc = np.stack([(xx[m] + 0.5) * cell, (H - yy[m] - 0.5) * cell], -1).astype(np.float32)
    kx = np.arange(0, W * cell + KNOT_SPACING_KM, KNOT_SPACING_KM)
    ky = np.arange(0, H * cell + KNOT_SPACING_KM, KNOT_SPACING_KM)
    knots = np.array([(a, b) for a in kx for b in ky], np.float32)
    Y = np.concatenate([y[:, m].T, ((z == 1) & anchor)[m][:, None]], 1).astype(np.int8)
    M = np.concatenate([cover[:, m].T, anchor[m][:, None]], 1)
    sp = split[m]
    basis = rbf_basis(loc, knots, KNOT_SPACING_KM)       # bandwidth = knot spacing (3 km)
    return dict(Xt=Xt, Xa=Xa, loc=loc, knots=knots, basis=basis, Y=Y, M=M, z=z[m], d_pub_km=dist["public"][m].astype(np.float32),
                tr=sp == 0, va=sp == 1, te=sp == 2)


def run_one(seed, scen, steps, restarts, grid):
    D = prepare(seed, scen, grid)
    Xt, Xa, tr, va, te = D["Xt"], D["Xa"], D["tr"], D["va"], D["te"]
    Xa_sp = np.concatenate([Xa, D["basis"]], 1)
    X_free = np.concatenate([Xt, Xa[:, :3]], 1)                  # Stage 3 scenario B: roads enter psi freely
    X_cut = np.concatenate([Xt, D["d_pub_km"][:, None]], 1)       # short-range road-cut term only
    X_terr = Xt
    fixed = [None, None, None, 0.99]
    n_tr = int(tr.sum())

    def ensemble(make, X_site, X_obs):
        T = lambda a: torch.tensor(a[tr]); V = lambda a: torch.tensor(a[va])
        loss_on = lambda mm, S: mm.loss(S(X_site), S(X_obs), S(D["Y"]), S(D["M"]))
        preds, extra = [], []
        for r in range(restarts):
            torch.manual_seed(1000 * seed + r)
            mm = make()
            train(mm, lambda: loss_on(mm, T), val_loss_fn=lambda: loss_on(mm, V),
                  steps=steps, seed=1000 * seed + r)
            preds.append(mm.predict(torch.tensor(X_site[te])).numpy())
            if isinstance(mm.psi_head, PsiWithRoadCut):
                with torch.no_grad():
                    extra.append((float(torch.nn.functional.softplus(mm.psi_head.a)), float(mm.psi_head.lam())))
            if isinstance(mm.p_head, SpatialMonotoneDetection):
                with torch.no_grad():
                    f = mm.p_head.field(torch.tensor(D["basis"])).numpy()
                extra.append(("field_sd", [float(v) for v in f.std(0)]))
        return np.mean(preds, 0), extra

    spatial = lambda: SpatialMonotoneDetection(SIGNS, 3, D["basis"].shape[1], n_train=n_tr)
    from models import MonotoneDetection
    mono = lambda: MonotoneDetection(SIGNS, 3)
    variants = {}
    if scen == "B":
        variants["FULL + spatial"] = (lambda: OccupancyNet(X_free.shape[1], 0, 4, fixed_p=fixed, p_head=spatial()),
                                      X_free, Xa_sp)
        variants["FULL + short-range road"] = (
            lambda: OccupancyNet(X_cut.shape[1], 0, 4, fixed_p=fixed, p_head=mono(),
                                 psi_head=PsiWithRoadCut(Xt.shape[1])), X_cut, Xa)
        variants["FULL + spatial + short-range road"] = (
            lambda: OccupancyNet(X_cut.shape[1], 0, 4, fixed_p=fixed, p_head=spatial(),
                                 psi_head=PsiWithRoadCut(Xt.shape[1])), X_cut, Xa_sp)
    else:
        variants["FULL + spatial"] = (lambda: OccupancyNet(X_terr.shape[1], 0, 4, fixed_p=fixed, p_head=spatial()),
                                      X_terr, Xa_sp)
    out, info = {}, {}
    zte, dte = D["z"][te], D["d_pub_km"][te]
    for name, (make, Xs, Xo) in variants.items():
        p, extra = ensemble(make, Xs, Xo)
        out[name] = metrics(p, zte, dte)
        out[name]["abs_log_ratio"] = float(abs(np.log(out[name]["remote_near_ratio"])))
        info[name] = extra
    out["_learned"] = info
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed-list", default=None, help="e.g. 0,1,2 (default: range(--seeds))")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    grid = load_grid()
    seeds = [int(v) for v in args.seed_list.split(",")] if args.seed_list else list(range(args.seeds))
    tag = args.scenario + ("_" + "".join(map(str, seeds)) if args.seed_list else "")
    res = {}
    for s in seeds:
        t0 = time.time()
        res[s] = run_one(s, args.scenario, args.steps, args.restarts, grid)
        print(f"[scenario {args.scenario}] seed {s}: {time.time() - t0:.0f}s", flush=True)
        (OUT / f"raw_results_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print("finished", flush=True)


if __name__ == "__main__":
    main()
