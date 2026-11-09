"""
Pre-registered prediction for a new earthquake, written BEFORE any model comparison is run.

Rule (fixed from Iburi and Gorkha, Stage 8): measure the interaction gap of the TRUE landslide pattern =
AUC(flexible GBM) - AUC(additive spline GAM), terrain features only, same 5 block splits as the experiments.
  gap > 0.03          -> predict the occupancy model beats the best Steger variant (Iburi: +0.058)
  gap < 0.015         -> predict Steger is as good or better                      (Gorkha: 0.000)
  0.015 <= gap <= 0.03 -> no confident prediction
Writes results_q3/<name>_prediction.json with a UTC timestamp.
Usage: python q3_preregister.py <grid.npz> <name>
"""
import datetime
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


def main(path, name):
    grid = load_grid(path)
    rows = []
    for seed in range(5):
        D, _ = build(seed, "A", grid)
        X, z, tr, te = D["Xt"], D["z"], D["tr"] | D["va"], D["te"]
        a = lambda m: roc_auc_score(z[te], m.fit(X[tr], z[tr]).predict_proba(X[te])[:, 1])
        glm = a(LogisticRegression(max_iter=2000))
        gam = a(make_pipeline(SplineTransformer(n_knots=6, degree=3), LogisticRegression(max_iter=3000)))
        gbm = a(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed))
        rows.append(dict(glm=glm, gam=gam, gbm=gbm))
        print(name, seed, {k: round(v, 3) for k, v in rows[-1].items()}, flush=True)
    gap = float(np.mean([r["gbm"] - r["gam"] for r in rows]))
    nonlin = float(np.mean([r["gbm"] - r["glm"] for r in rows]))
    pred = ("occupancy model beats best Steger" if gap > 0.03 else
            "Steger as good or better" if gap < 0.015 else "no confident prediction")
    out = dict(name=name, interaction_gap=gap, nonlinearity_gap=nonlin, per_draw=rows, prediction=pred,
               rule="gap>0.03 ours wins; gap<0.015 Steger >= ours; otherwise no prediction",
               written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (HERE / "results_q3").mkdir(exist_ok=True)
    (HERE / f"results_q3/{name}_prediction.json").write_text(json.dumps(out, indent=1))
    print(f"{name}: interaction gap {gap:+.3f}, nonlinearity gap {nonlin:+.3f} -> PREDICTION: {pred}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
