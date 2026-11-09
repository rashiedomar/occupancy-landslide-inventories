"""
Revision check of the probability claims. Re-fits the headline models with the code and seeds of
run_headline.py and evaluates, on test blocks, against the reference inventory:
  - overall rate ratio, Brier score, log loss, expected calibration error (10 equal-frequency bins)
    and calibration slope/intercept;
  - the same for each baseline after a simple correction using the anchor area: divide by the share of anchor
    landslide cells recorded by any inventory (the label-frequency correction of Elkan and Noto, 2008).
This asks whether the occupancy model's rate advantage needs the likelihood or only the anchor data.
Writes results_rev/calib_<site>_<scenario>_<seed>.json. Usage: python rev_calibration.py <site> <scenario> <seed>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "toy_simulation"))
sys.path.insert(0, str(HERE))
from models import BinaryClassifier, MonotoneDetection, OccupancyNet, train  # noqa: E402
from run_headline import build  # noqa: E402
from run_iburi import load_grid, metrics  # noqa: E402
from run_iburi_v3 import SIGNS  # noqa: E402

P = HERE.parent / "data/processed"
OUT = HERE / "results_rev"; OUT.mkdir(exist_ok=True)
SITES = {"iburi": P / "iburi_grid.npz", "gorkha": P / "gorkha_grid.npz", "haiti": P / "haiti_grid.npz"}


def scores(p, z, n_bins=10):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    mp = np.array([p[b].mean() for b in bins]); mz = np.array([z[b].mean() for b in bins])
    lg = np.log(p / (1 - p))[:, None]
    rc = LogisticRegression(C=1e6, max_iter=2000).fit(lg, z)
    return dict(auc=roc_auc_score(z, p), rate_ratio=float(p.mean() / z.mean()), brier=brier_score_loss(z, p),
                log_loss=log_loss(z, p), ece=float(np.mean(np.abs(mp - mz))),
                cal_slope=float(rc.coef_[0, 0]), cal_intercept=float(rc.intercept_[0]),
                bin_pred=mp.tolist(), bin_true=mz.tolist())


def main(site, scen, seed, restarts=3, steps=3000):
    torch.set_num_threads(1)
    D, info = build(seed, scen, load_grid(SITES[site]))
    Xt, Xa, Y, M, tr, va, te = D["Xt"], D["Xa"], D["Y"], D["M"], D["tr"], D["va"], D["te"]
    z = D["z"]
    T = lambda a: torch.tensor(a[tr]); V = lambda a: torch.tensor(a[va])
    y_any = Y.max(1)
    X_all = np.concatenate([Xt, Xa], 1); nt = Xt.shape[1]
    X_neutral = X_all[te].copy(); X_neutral[:, nt:] = 0.0
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
    oracle = ens(lambda: BinaryClassifier(nt), lambda mm, S: mm.loss(S(Xt), S(z)))
    preds = {"standard": net(X_all[te]), "steger_mlp": net(X_neutral), "steger_glm": glm(X_neutral),
             "ours": full(Xt[te]), "oracle": oracle(Xt[te])}

    anchor = M[:, 3].astype(bool)
    n_slides = int(z[anchor].sum())
    recorded = int(Y[anchor & (z == 1), :3].max(1).sum()) if n_slides else 0
    c = recorded / n_slides if n_slides else float("nan")
    out = {"_anchor": dict(cells=int(anchor.sum()), landslide_cells=n_slides, recorded=recorded, label_freq=c),
           "_info": info}
    for k, p in preds.items():
        out[k] = scores(p, z[te])
        mt = metrics(p, z[te], D["d_pub"][te])
        out[k]["auc_remote"] = mt["auc_remote_quarter"]
        out[k]["road_bias"] = float(abs(np.log(mt["remote_near_ratio"])))
    for k in ["standard", "steger_mlp", "steger_glm"]:
        if c > 0:
            out[k + "_anchor_scaled"] = scores(np.clip(preds[k] / c, 0, 1), z[te])
    (OUT / f"calib_{site}_{scen}_{seed}.json").write_text(json.dumps(out, indent=1, default=float))
    print(site, scen, seed, "done", {k: round(out[k]["rate_ratio"], 2) for k in out if not k.startswith("_")}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
