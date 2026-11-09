"""
Revision check of the method-selection rule.

For each site and transplant world, measures the interaction gap AUC(GBM) - AUC(additive GAM) on terrain
features, same 5 block splits as the experiments, in two ways:
  truth   : trained and scored on the reference inventory (how the Haiti prediction was made)
  records : trained and scored on the union of the biased records only (what a practitioner can compute)
Writes results_rev/rule_checks_<world>.json. Usage: python rev_rule_checks.py <world> [scenario]
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
from run_iburi import _z, load_grid  # noqa: E402

P = HERE.parent / "data/processed"
OUT = HERE / "results_rev"; OUT.mkdir(exist_ok=True)
SITES = {"iburi": P / "iburi_grid.npz", "gorkha": P / "gorkha_grid.npz", "haiti": P / "haiti_grid.npz"}


def sig(x):
    return 1 / (1 + np.exp(-x))


def calibrate(logit, rate):
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if sig(logit + mid).mean() > rate else (mid, hi)
    return logit + mid


def interactions(X, names):
    c = {n: X[:, i] for i, n in enumerate(names)}
    h = (c["slope_mean"] * -c["curvature"] + c["relief"] * c["aspect_cos"] + c["tpi_150m"] * c["elev"]
         + 1.5 * (np.exp(-(c["slope_mean"] - 0.8) ** 2) - 0.6))
    h = h - np.polyval(np.polyfit(np.arange(len(h)), h, 0), 0)
    A = np.c_[np.ones(len(h)), X]
    h = h - A @ np.linalg.lstsq(A, h, rcond=None)[0]
    return h / h.std()


def world_truth(world, grid):
    """Rebuild a transplant world's truth with the recipes and seeds of q2_flip_worlds / q2_transplant."""
    d, m, terrain, dist = grid
    names = sorted(terrain)
    Xt = np.stack([_z(terrain[k], m)[m] for k in names], -1)
    z = d["z"][m].astype(int); rate = z.mean()
    tag = world.split("_")[1]
    if tag == "H1":                                 # recipe of q2_proxy_world.py
        from scipy import ndimage
        lg0 = LogisticRegression(max_iter=3000).fit(Xt, z).decision_function(Xt)
        field = ndimage.gaussian_filter(np.random.default_rng(11).standard_normal(m.shape), 20)
        H = _z(0.7 * _z(np.log1p(dist["public"]), m) + 0.7 * _z(field, m), m)[m]
        lg = calibrate(lg0 + 1.5 * H, rate)
        return (np.random.default_rng(123).random(len(lg)) < sig(lg)).astype(np.int8)
    if tag in ("L", "N"):
        lg0 = LogisticRegression(max_iter=3000).fit(Xt, z).decision_function(Xt)
        if tag == "L":
            lg = calibrate(lg0, rate)
        else:
            lg = calibrate(lg0 + NSCALE[world] * interactions(Xt, names), rate)
        return (np.random.default_rng(123).random(len(lg)) < sig(lg)).astype(np.int8)
    if tag == "W1a":
        mdl = make_pipeline(SplineTransformer(n_knots=8, degree=3), LogisticRegression(max_iter=4000)); X = Xt
    else:
        hgb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                             min_samples_leaf=50, random_state=0)
        mdl = hgb
        if tag == "W1":
            X = Xt
        elif tag == "W2":
            Xa = np.stack([_z(a, m)[m] for a in [np.log1p(dist["major"]), np.log1p(dist["public"]),
                                                 np.log1p(dist["track"]), d["built_density"]]], -1)
            X = np.c_[Xt, Xa]
        elif tag == "W3":
            X = Xt[:, [i for i, n in enumerate(names) if not n.startswith("epi")]]
    p = np.clip(mdl.fit(X, z).predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    lg = calibrate(np.log(p / (1 - p)), rate)
    return (np.random.default_rng(321).random(len(lg)) < sig(lg)).astype(np.int8)


NSCALE = {"iburi_N": 1.5, "gorkha_N": 1.5}   # overwritten from the target-gap search in main()


def search_scale(world, grid):
    """Same search as q2_flip_worlds: scale of the interaction term that hits a GBM-GLM gap of 0.09."""
    d, m, terrain, _ = grid
    names = sorted(terrain)
    X = np.stack([_z(terrain[k], m)[m] for k in names], -1)
    z = d["z"][m].astype(int); rate = z.mean()
    lg0 = LogisticRegression(max_iter=3000).fit(X, z).decision_function(X)
    h = interactions(X, names)
    idx = np.random.default_rng(0).permutation(len(z)); k = int(0.7 * len(z)); tr, te = idx[:k], idx[k:]
    best = None
    for s in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
        lg = calibrate(lg0 + s * h, rate)
        zz = (np.random.default_rng(123).random(len(lg)) < sig(lg)).astype(int)
        glm = LogisticRegression(max_iter=2000).fit(X[tr], zz[tr])
        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X[tr], zz[tr])
        g = (roc_auc_score(zz[te], gbm.predict_proba(X[te])[:, 1]) - roc_auc_score(zz[te], glm.predict_proba(X[te])[:, 1]))
        if best is None or abs(g - 0.09) < abs(best[1] - 0.09):
            best = (s, g)
    return best[0]


def gap(X, y_tr, y_te, tr, te, seed):
    a = lambda mdl: roc_auc_score(y_te, mdl.fit(X[tr], y_tr).predict_proba(X[te])[:, 1])
    gam = a(make_pipeline(SplineTransformer(n_knots=6, degree=3), LogisticRegression(max_iter=3000)))
    gbm = a(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed))
    return dict(gam=gam, gbm=gbm, gap=gbm - gam)


def real_records():
    """Gap on the three real Gorkha inventories (plus anchor), trained and scored on those records only."""
    from q1_run_real import GRID, OBS, setup
    grid = load_grid(GRID); obs = dict(np.load(OBS))
    rows = []
    for seed in range(5):
        D = setup(seed, grid, obs)
        X, Y, tr, te = D["Xt"], D["Y"], D["tr"] | D["va"], D["te"]
        yany = Y.max(1)
        rows.append(dict(seed=seed, records=gap(X, yany[tr], yany[te], tr, te, seed),
                         truth=gap(X, D["z"][tr], D["z"][te], tr, te, seed)))
    out = dict(world="gorkha_real_inventories", rows=rows,
               truth_gap=float(np.mean([x["truth"]["gap"] for x in rows])),
               records_gap=float(np.mean([x["records"]["gap"] for x in rows])))
    (OUT / "rule_checks_gorkha_real.json").write_text(json.dumps(out, indent=1))
    print(out["world"], out["truth_gap"], out["records_gap"], flush=True)


def main(world, scen="A"):
    if world == "real":
        return real_records()
    site = world.split("_")[0]
    grid = load_grid(SITES[site])
    d, m, terrain, dist = grid
    truth_note = "reference inventory"
    if "_" in world:
        if world.endswith("_N"):
            NSCALE[world] = search_scale(world, grid)
        zz = world_truth(world, grid)
        zf = np.zeros_like(d["z"]); zf[m] = zz
        raw = {k: d[k] for k in d.files}; raw["z"] = zf
        grid = (raw, m, terrain, dist)
        truth_note = "transplant world"
    rows = []
    for seed in range(5):
        D, info = build(seed, scen, grid)
        X, z, Y, tr, te = D["Xt"], D["z"], D["Y"], D["tr"] | D["va"], D["te"]
        yany = Y.max(1)
        t = gap(X, z[tr], z[te], tr, te, seed)
        r = gap(X, yany[tr], yany[te], tr, te, seed)
        rows.append(dict(seed=seed, truth=t, records=r, frac_recorded=info["frac_recorded"]))
        print(world, scen, seed, round(t["gap"], 3), round(r["gap"], 3), flush=True)
    out = dict(world=world, scenario=scen, truth_note=truth_note, rows=rows,
               truth_gap=float(np.mean([x["truth"]["gap"] for x in rows])),
               records_gap=float(np.mean([x["records"]["gap"] for x in rows])))
    (OUT / f"rule_checks_{world}_{scen}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "A")
