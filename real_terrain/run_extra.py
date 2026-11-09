"""
Two extra stress tests on real Iburi terrain.

D  "planted road cuts": we ADD synthetic landslides near roads with a known effect
   (p_extra = 0.25 * exp(-d_road / 0.12 km) * sigmoid(2 * slope_z)), so the truth now contains a genuine
   short-range road effect. Then the same biased reporting as Stage 2/3 is simulated on the new truth.
   Question: can a model capture the REAL road effect without absorbing the REPORTING bias?
E  "dependent records": the news catalog copies half of the agency's records (violates the
   conditional-independence assumption of the occupancy model).

Usage: python run_extra.py --scenario D   (or E)
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
from run_iburi import _sig, _z, block_split, load_grid, metrics, simulate_inventories  # noqa: E402
from run_iburi_v3 import ACCESS, SIGNS, pick_anchor  # noqa: E402

OUT = HERE / "results_extra"; OUT.mkdir(exist_ok=True)
PLANT = dict(strength=0.25, range_km=0.12)


def build(seed, scen, grid):
    d, m, terrain, dist = grid
    rng = np.random.default_rng(seed)
    z = d["z"].astype(np.int8)
    built = d["built_density"]
    if scen == "D":
        prng = np.random.default_rng(50_000 + seed)
        p_extra = PLANT["strength"] * np.exp(-dist["public"] / PLANT["range_km"]) * _sig(2 * _z(terrain["slope_mean"], m))
        added = (prng.random(z.shape) < p_extra) & m & (z == 0)
        z = (z | added).astype(np.int8)
    y, p_true, cover, cloud = simulate_inventories(z, m, terrain, dist, built, rng)
    if scen == "E":
        erng = np.random.default_rng(60_000 + seed)
        copy = (y[1] == 1) & (erng.random(z.shape) < 0.5)
        y[0] = (y[0] | copy).astype(np.int8)        # news reports re-publish agency records
    split = block_split(m, rng)
    anchor = pick_anchor(split, m, np.random.default_rng(10_000 + seed))
    Xt = np.stack([_z(terrain[k], m)[m] for k in sorted(terrain)], -1).astype(np.float32)
    acc = {"log_d_major": np.log1p(dist["major"]), "log_d_public": np.log1p(dist["public"]),
           "log_d_track": np.log1p(dist["track"]), "built": built, "cloud": cloud}
    Xa = np.stack([_z(acc[k], m)[m] for k in ACCESS], -1).astype(np.float32)
    Y = np.concatenate([y[:, m].T, ((z == 1) & anchor)[m][:, None]], 1).astype(np.int8)
    M = np.concatenate([cover[:, m].T, anchor[m][:, None]], 1)
    sp = split[m]
    info = {"n_added": int(added.sum()) if scen == "D" else 0, "n_true": int(z[m].sum()),
            "frac_recorded": float(Y[:, :3].max(1)[z[m] == 1].mean())}
    return dict(Xt=Xt, Xa=Xa, Y=Y, M=M, z=z[m], d_pub=dist["public"][m].astype(np.float32),
                tr=sp == 0, va=sp == 1, te=sp == 2), info


def lift_ratio(pred, z, d):
    """(rate within 150 m of a road / rate at 150-500 m) for the model, divided by the same for the truth.
    1 = the model reproduces the true near-road excess."""
    near, mid = d < 0.15, (d >= 0.15) & (d < 0.5)
    return float((pred[near].mean() / pred[mid].mean()) / (z[near].mean() / z[mid].mean()))


def run_one(seed, scen, grid, restarts=3, steps=3000):
    D, info = build(seed, scen, grid)
    Xt, Xa, Y, M, tr, va, te = D["Xt"], D["Xa"], D["Y"], D["M"], D["tr"], D["va"], D["te"]
    T = lambda a: torch.tensor(a[tr]); V = lambda a: torch.tensor(a[va])
    y_any = Y.max(1)
    X_free = np.concatenate([Xt, Xa[:, :3]], 1)
    X_cut = np.concatenate([Xt, D["d_pub"][:, None]], 1)
    fixed = [None, None, None, 0.99]

    def ens(make, loss_on, predict):
        outs, ms = [], []
        for r in range(restarts):
            torch.manual_seed(1000 * seed + r)
            mm = make()
            train(mm, lambda: loss_on(mm, T), val_loss_fn=lambda: loss_on(mm, V), steps=steps, seed=1000 * seed + r)
            outs.append(predict(mm)); ms.append(mm)
        return np.mean(outs, 0), ms

    preds, learned = {}, {}
    X_all = np.concatenate([Xt, Xa], 1)
    nt = Xt.shape[1]
    X_neutral = X_all[te].copy(); X_neutral[:, nt:] = 0.0
    std, std_ms = ens(lambda: BinaryClassifier(X_all.shape[1]), lambda mm, S: mm.loss(S(X_all), S(y_any)),
                      lambda mm: mm.predict(torch.tensor(X_all[te])).numpy())
    preds["Standard (terrain + access)"] = std
    preds["Steger-MLP (access = mean)"] = np.mean([mm.predict(torch.tensor(X_neutral)).numpy() for mm in std_ms], 0)

    def occ(name, Xs, psi=None):
        make = lambda: OccupancyNet(Xs.shape[1], 0, 4, fixed_p=fixed, p_head=MonotoneDetection(SIGNS, 3),
                                    psi_head=psi() if psi else None)
        preds[name], ms = ens(make, lambda mm, S: mm.loss(S(Xs), S(Xa), S(Y), S(M)),
                              lambda mm: mm.predict(torch.tensor(Xs[te])).numpy())
        if psi:
            with torch.no_grad():
                learned[name] = [(float(torch.nn.functional.softplus(mm.psi_head.a)), float(mm.psi_head.lam()))
                                 for mm in ms]

    occ("OURS: terrain-only susceptibility", Xt)
    occ("OURS: + short-range road term", X_cut, psi=lambda: PsiWithRoadCut(Xt.shape[1]))
    if scen == "D":
        occ("OURS: roads enter susceptibility freely", X_free)
    X_or = X_cut if scen == "D" else Xt
    preds["Oracle"], _ = ens(lambda: BinaryClassifier(X_or.shape[1]), lambda mm, S: mm.loss(S(X_or), S(D["z"])),
                             lambda mm: mm.predict(torch.tensor(X_or[te])).numpy())

    zte, dte = D["z"][te], D["d_pub"][te]
    out = {}
    for k, p in preds.items():
        out[k] = metrics(p, zte, dte)
        out[k]["abs_log_ratio"] = float(abs(np.log(out[k]["remote_near_ratio"])))
        out[k]["near_road_lift_ratio"] = lift_ratio(p, zte, dte)
    out["_learned_roadcut"] = learned
    out["_info"] = info
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    grid = load_grid()
    res = {}
    for s in range(args.seeds):
        t0 = time.time()
        res[s] = run_one(s, args.scenario, grid)
        print(f"[{args.scenario}] draw {s}: {time.time() - t0:.0f}s  {res[s]['_info']}", flush=True)
        (OUT / f"raw_results_{args.scenario}.json").write_text(json.dumps(res, indent=1, default=float))
    print("finished", flush=True)


if __name__ == "__main__":
    main()
