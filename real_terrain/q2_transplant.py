"""
Q2 decomposition test using Iburi's OWN signal (no hand-built drivers).
W1: truth psi = flexible fit (GBM) of the real Iburi inventory on TERRAIN only
    -> keeps Iburi's real nonlinear terrain structure, drops anything road variables proxy for.
W2: truth psi = flexible fit of the real inventory on TERRAIN + ACCESS (road distances, built-up)
    -> also keeps the genuine signal that road variables carry as proxies for unmeasured drivers.
Same terrain, roads, prevalence and reporting simulation. Proxy hypothesis predicts:
Steger >= ours in W1, ours > Steger in W2.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run_iburi import _z, load_grid  # noqa: E402
from q2_flip_worlds import calibrate, sig  # noqa: E402

P = HERE.parent / "data/processed"
d, m, terrain, dist = load_grid(P / "iburi_grid.npz")
raw = dict(np.load(P / "iburi_grid.npz", allow_pickle=True))
Xt = np.stack([_z(terrain[k], m)[m] for k in sorted(terrain)], -1)
Xa = np.stack([_z(a, m)[m] for a in [np.log1p(dist["major"]), np.log1p(dist["public"]),
                                      np.log1p(dist["track"]), d["built_density"]]], -1)
z = d["z"][m].astype(int)
rate = z.mean()
def base_worlds():
  for name, X in [("W1", Xt), ("W2", np.c_[Xt, Xa])]:
      gbm = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=50, random_state=0).fit(X, z)
      p = np.clip(gbm.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
      lg = calibrate(np.log(p / (1 - p)), rate)
      zz = (np.random.default_rng(321).random(len(lg)) < sig(lg)).astype(np.int8)
      print(f"{name}: prevalence {zz.mean():.3f} (real {rate:.3f}); corr(truth psi, log road dist) "
            f"{np.corrcoef(sig(lg), Xa[:, 1])[0, 1]:+.2f}", flush=True)
      new = dict(raw)
      zf = np.zeros_like(raw["z"]); zf[m] = zz
      new["z"] = zf
      np.savez_compressed(P / f"flip/iburi_{name}.npz", **new)


def extra_worlds():
    """W3: real-pattern fit WITHOUT epicentral distance; W1a: ADDITIVE (spline GAM) fit with all terrain."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer
    names = sorted(terrain)
    no_epi = [i for i, n in enumerate(names) if not n.startswith("epi")]
    fits = {"W3": (HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                                  min_samples_leaf=50, random_state=0), Xt[:, no_epi]),
            "W1a": (make_pipeline(SplineTransformer(n_knots=8, degree=3), LogisticRegression(max_iter=4000)), Xt)}
    for name, (mdl, X) in fits.items():
        p = np.clip(mdl.fit(X, z).predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
        lg = calibrate(np.log(p / (1 - p)), rate)
        zz = (np.random.default_rng(321).random(len(lg)) < sig(lg)).astype(np.int8)
        print(f"{name}: prevalence {zz.mean():.3f}", flush=True)
        new = dict(raw); zf = np.zeros_like(raw["z"]); zf[m] = zz; new["z"] = zf
        np.savez_compressed(P / f"flip/iburi_{name}.npz", **new)


if __name__ == "__main__":
    (extra_worlds if len(sys.argv) > 1 and sys.argv[1] == "extra" else base_worlds)()
