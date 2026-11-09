"""
Q2: why does the occupancy model beat Steger neutralisation on Iburi but not on Gorkha?

Hypothesis. Steger's fix requires the bias variables to be neutralisable:
  * in a GLM this is clean, but a GLM cannot represent nonlinear/interacting susceptibility;
  * in a flexible model (MLP) the susceptibility can be nonlinear, but setting access features to a
    constant creates (terrain, access) combinations the model never saw -> off-distribution inputs.
The occupancy model separates "where landslides happen" and "where they are recorded" structurally,
so it needs neither. Prediction: the more nonlinear the true susceptibility, and the more off-distribution
the neutralised inputs, the larger our advantage. Iburi should score higher on both than Gorkha.

Measures (on each earthquake, full real inventory, same 5 block splits as the experiments):
  1. nonlinearity  : AUC of oracle GLM vs additive spline GAM vs MLP (all on terrain only).
  2. off-manifold  : how easily a classifier tells real (terrain, access) rows from neutralised rows
                     (AUC 0.5 = indistinguishable).
  3. per-draw link : across the 20 draws, does our AUC advantage over Steger track these measures?
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "toy_simulation"))
sys.path.insert(0, str(HERE))
from run_headline import build  # noqa: E402
from run_iburi import load_grid  # noqa: E402

GRIDS = {"Iburi": HERE.parent / "data/processed/iburi_grid.npz",
         "Gorkha": HERE.parent / "data/processed/gorkha_grid.npz"}


def main():
    out = {}
    for name, path in GRIDS.items():
        grid = load_grid(path)
        rows = []
        for seed in range(5):
            D, _ = build(seed, "A", grid)
            Xt, Xa, z, tr, te = D["Xt"], D["Xa"], D["z"], D["tr"] | D["va"], D["te"]
            glm = LogisticRegression(max_iter=2000).fit(Xt[tr], z[tr])
            gam = make_pipeline(SplineTransformer(n_knots=6, degree=3),
                                LogisticRegression(max_iter=3000, C=1.0)).fit(Xt[tr], z[tr])
            gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                 random_state=seed).fit(Xt[tr], z[tr])
            auc = lambda m: roc_auc_score(z[te], m.predict_proba(Xt[te])[:, 1])
            # off-manifold: real rows vs neutralised rows (access set to its mean = 0 after standardising)
            real = np.concatenate([Xt, Xa], 1)
            neut = real.copy(); neut[:, Xt.shape[1]:] = 0.0
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(real))
            half = len(idx) // 2
            Xd = np.concatenate([real[idx[:half]], neut[idx[half:]]])
            yd = np.r_[np.zeros(half), np.ones(len(idx) - half)]
            p = rng.permutation(len(Xd)); k = int(0.7 * len(p))
            disc = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xd[p[:k]], yd[p[:k]])
            off = roc_auc_score(yd[p[k:]], disc.predict_proba(Xd[p[k:]])[:, 1])
            # entanglement: how well terrain predicts access (R^2 of a GBM on log road distance)
            from sklearn.ensemble import HistGradientBoostingRegressor
            reg = HistGradientBoostingRegressor(max_iter=200, random_state=seed).fit(Xt[tr], Xa[tr, 1])
            r2 = 1 - np.mean((reg.predict(Xt[te]) - Xa[te, 1]) ** 2) / np.var(Xa[te, 1])
            rows.append(dict(glm=auc(glm), gam=auc(gam), gbm=auc(gbm), off_manifold_auc=off, entangle_r2=r2))
            print(name, seed, {k: round(v, 3) for k, v in rows[-1].items()}, flush=True)
        out[name] = rows
    (HERE / "results_q2").mkdir(exist_ok=True)
    (HERE / "results_q2/mechanism_measures.json").write_text(json.dumps(out, indent=1))
    for name, rows in out.items():
        m = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        print(f"{name}: GLM {m['glm']:.3f} | additive GAM {m['gam']:.3f} | flexible GBM {m['gbm']:.3f} | "
              f"nonlinearity gap (GBM-GLM) {m['gbm'] - m['glm']:+.3f}, interaction gap (GBM-GAM) {m['gbm'] - m['gam']:+.3f} | "
              f"off-manifold AUC {m['off_manifold_auc']:.3f} | terrain->road-distance R2 {m['entangle_r2']:.2f}")


if __name__ == "__main__":
    main()
