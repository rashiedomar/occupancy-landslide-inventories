"""
Revision check: neutralisation in a generalised additive model, the model class of Steger et al. (2021).
Additive splines on terrain + access features, fitted to the union of records; access features are set to their
mean for prediction. Same draws and splits as run_headline.py.
Writes results_rev/gam_<world>_<scenario>.json. Usage: python rev_gam_baseline.py <world> <scenario>
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "toy_simulation"))
sys.path.insert(0, str(HERE))
from rev_rule_checks import NSCALE, OUT, SITES, search_scale, world_truth  # noqa: E402
from run_headline import build  # noqa: E402
from run_iburi import load_grid, metrics  # noqa: E402


def main(world, scen):
    grid = load_grid(SITES[world.split("_")[0]])
    d, m, terrain, dist = grid
    if "_" in world:
        if world.endswith("_N"):
            NSCALE[world] = search_scale(world, grid)
        zf = np.zeros_like(d["z"]); zf[m] = world_truth(world, grid)
        raw = {k: d[k] for k in d.files}; raw["z"] = zf
        grid = (raw, m, terrain, dist)
    rows = []
    for seed in range(5):
        D, _ = build(seed, scen, grid)
        Xt, Xa, Y, z, tr, te = D["Xt"], D["Xa"], D["Y"], D["z"], D["tr"] | D["va"], D["te"]
        X = np.concatenate([Xt, Xa], 1)
        mdl = make_pipeline(SplineTransformer(n_knots=6, degree=3), LogisticRegression(max_iter=3000))
        mdl.fit(X[tr], Y.max(1)[tr])
        Xn = X[te].copy(); Xn[:, Xt.shape[1]:] = 0.0
        p = mdl.predict_proba(Xn)[:, 1]
        r = metrics(p, z[te], D["d_pub"][te])
        r["abs_log_ratio"] = float(abs(np.log(r["remote_near_ratio"])))
        rows.append(r)
        print(world, scen, seed, round(r["auc_true_landslides"], 3), flush=True)
    (OUT / f"gam_{world}_{scen}.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
