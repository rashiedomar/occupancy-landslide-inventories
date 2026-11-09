"""
The missing competitor: Steger-style "fit with bias variables, neutralise them at prediction".

Steger et al. (2017, 2021) show that leaving bias-describing predictors (e.g. road distance) OUT of the
model confounds the other effects, and instead propose to fit the model WITH them and then hold them
at a constant when mapping, so they cannot shape the prediction. We test three versions:

  Steger-GLM            : logistic regression (Steger's setting); the constant does not change ranking.
  Steger-MLP (mean)     : same MLP as "Standard (terrain + access)", access features set to their mean.
  Steger-MLP (best obs) : access features set to 'best observation' conditions (5th/95th percentile:
                          next to roads, built-up, clear sky).

Same records, draws, splits and 3-restart ensembling as Stages 3-4. Scenarios A (identical data to B for
these models) and C.
Usage: python run_steger_baseline.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "toy_simulation"))
sys.path.insert(0, str(HERE))
from models import BinaryClassifier, train  # noqa: E402
from run_iburi import load_grid, metrics  # noqa: E402
from run_iburi_v3 import ACCESS  # noqa: E402
from run_iburi_v4 import prepare  # noqa: E402

torch.set_num_threads(2)
OUT = HERE / "results_steger"; OUT.mkdir(exist_ok=True)
# direction in which each access feature makes recording MORE likely (for 'best observation')
BEST = {"log_d_major": "low", "log_d_public": "low", "log_d_track": "low", "built": "high", "cloud": "low"}


def run_one(seed, scen, grid, restarts=3, steps=3000):
    D = prepare(seed, "A" if scen == "A" else "C", grid)
    Xt, Xa, tr, va, te = D["Xt"], D["Xa"], D["tr"], D["va"], D["te"]
    X = np.concatenate([Xt, Xa], 1)
    y_any = D["Y"].max(1)
    nt = Xt.shape[1]
    X_mean = X[te].copy(); X_mean[:, nt:] = 0.0                       # features are standardised
    best = np.array([np.percentile(Xa[:, j], 5 if BEST[f] == "low" else 95) for j, f in enumerate(ACCESS)],
                    np.float32)
    X_best = X[te].copy(); X_best[:, nt:] = best

    def ensemble(depth):
        T = lambda a: torch.tensor(a[tr]); V = lambda a: torch.tensor(a[va])
        ms = []
        for r in range(restarts):
            torch.manual_seed(1000 * seed + r)
            mm = BinaryClassifier(X.shape[1], depth=depth)
            train(mm, lambda: mm.loss(T(X), T(y_any)), val_loss_fn=lambda: mm.loss(V(X), V(y_any)),
                  steps=steps, seed=1000 * seed + r)
            ms.append(mm)
        return lambda Xp: np.mean([mm.predict(torch.tensor(Xp)).numpy() for mm in ms], 0)

    glm, net = ensemble(depth=0), ensemble(depth=2)
    preds = {"Steger-GLM (access neutralised)": glm(X_mean),
             "Steger-MLP (access = mean)": net(X_mean),
             "Steger-MLP (access = best observation)": net(X_best),
             "Standard (terrain + access), re-run": net(X[te])}
    out = {}
    for k, p in preds.items():
        out[k] = metrics(p, D["z"][te], D["d_pub_km"][te])
        out[k]["abs_log_ratio"] = float(abs(np.log(out[k]["remote_near_ratio"])))
    return out


def main():
    grid = load_grid()
    res = {"A": {}, "C": {}}
    for scen in ["A", "C"]:
        for s in range(5):
            t0 = time.time()
            res[scen][s] = run_one(s, scen, grid)
            print(f"[{scen}] draw {s}: {time.time() - t0:.0f}s", flush=True)
            (OUT / "raw_results.json").write_text(json.dumps(res, indent=1, default=float))
    compare(res)


def compare(res):
    v3 = {}
    for f in (HERE / "results_v3").glob("raw_results_*.json"):
        v3.update(json.loads(f.read_text()))
    v4 = {}
    for f in (HERE / "results_v4").glob("raw_results_B_*.json"):
        v4.update({int(k): v for k, v in json.loads(f.read_text()).items()})
    lines = []
    for scen in ["A", "C"]:
        rows = {"Standard (terrain + access)": [v3[scen][s]["Standard (terrain + access)"] for s in range(5)]}
        for k in res[scen][0]:
            rows[k] = [res[scen][s][k] for s in range(5)]
        rows["OURS: occupancy full (terrain-only psi)"] = [v3[scen][s]["Occupancy + monotone + anchor (FULL)"] for s in range(5)]
        if scen == "A":
            rows["OURS: + short-range road (scen. B data)"] = [v4[s]["FULL + short-range road"] for s in range(5)]
        rows["Oracle"] = [v3[scen][s]["Oracle (full real inventory)"] for s in range(5)]
        lines.append(f"\n=== Scenario {scen} {'(same data as B)' if scen == 'A' else '(stress test)'}: 5 draws ===")
        lines.append(f"{'model':44s}{'AUC':>16s}{'AUC remote 1/4':>18s}{'|log bias|':>16s}{'bias ratio med':>16s}")
        for k, rs in rows.items():
            f = lambda m: np.array([r[m] for r in rs])
            lines.append(f"{k:44s}{f('auc_true_landslides').mean():>9.3f} ± {f('auc_true_landslides').std():.3f}"
                         f"{f('auc_remote_quarter').mean():>11.3f} ± {f('auc_remote_quarter').std():.3f}"
                         f"{f('abs_log_ratio').mean():>9.2f} ± {f('abs_log_ratio').std():.2f}"
                         f"{np.median(f('remote_near_ratio')):>14.2f}")
        ours = rows["OURS: occupancy full (terrain-only psi)"]
        for k in [k for k in rows if k.startswith("Steger")]:
            a = sum(o["auc_true_landslides"] > b["auc_true_landslides"] for o, b in zip(ours, rows[k]))
            b_ = sum(o["abs_log_ratio"] < b["abs_log_ratio"] for o, b in zip(ours, rows[k]))
            lines.append(f"  OURS vs {k}: higher AUC in {a}/5 draws, less bias in {b_}/5 draws")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "summary.txt").write_text(txt)


if __name__ == "__main__":
    main()
