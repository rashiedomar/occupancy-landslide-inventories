"""
Q2 decisive test: add a HIDDEN driver to the truth on Gorkha terrain (where we lose to Steger).
The hidden driver H is a smooth spatial field correlated with road distance (as unmeasured geology / deposits /
land use are in Iburi). Truth logit = GLM(terrain fitted to real inventory) + s * H, with s tuned so that adding
access features to terrain gains ~ +0.05 AUC on the TRUE labels (Iburi's real value). Same prevalence.
Prediction: on this world the occupancy model should now beat Steger.
"""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run_iburi import _z, load_grid
from q2_flip_worlds import calibrate, sig
P = HERE.parent / "data/processed"

def access_gain(Xt, Xa, z, seed=0):
    r = np.random.default_rng(seed); idx = r.permutation(len(z)); k = int(0.7 * len(z)); tr, te = idx[:k], idx[k:]
    f = lambda X: roc_auc_score(z[te], HistGradientBoostingClassifier(max_iter=80, learning_rate=0.15, random_state=0).fit(X[tr], z[tr]).predict_proba(X[te])[:, 1])
    return f(np.c_[Xt, Xa]) - f(Xt)

d, m, terrain, dist = load_grid(P / "gorkha_grid.npz")
raw = dict(np.load(P / "gorkha_grid.npz", allow_pickle=True))
Xt = np.stack([_z(terrain[k], m)[m] for k in sorted(terrain)], -1)
acc = [np.log1p(dist["major"]), np.log1p(dist["public"]), np.log1p(dist["track"]), d["built_density"]]
Xa = np.stack([_z(a, m)[m] for a in acc], -1)
z_real = d["z"][m].astype(int); rate = z_real.mean()
glm_logit = LogisticRegression(max_iter=3000).fit(Xt, z_real).decision_function(Xt)
field = ndimage.gaussian_filter(np.random.default_rng(11).standard_normal(m.shape), 20)   # ~4 km smooth field
Hfull = 0.7 * _z(np.log1p(dist["public"]), m) + 0.7 * _z(field, m)
H = _z(Hfull, m)[m]
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "H1"
if VARIANT == "H2":      # road-specific: remove the part terrain can explain (flexibly)
    from sklearn.ensemble import HistGradientBoostingRegressor
    H = H - HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xt, H).predict(Xt)
    H = (H - H.mean()) / H.std()
print("corr(H, log road distance) =", round(float(np.corrcoef(H, Xa[:, 1])[0, 1]), 2),
      "| corr(H, elevation) =", round(float(np.corrcoef(H, Xt[:, sorted(terrain).index("elev")])[0, 1]), 2))
best = None
for s in ([1.5] if VARIANT == "H1" else [0.75, 1.25, 2.0]):
    lg = calibrate(glm_logit + s * H, rate)
    zz = (np.random.default_rng(123).random(len(lg)) < sig(lg)).astype(int)
    g = access_gain(Xt, Xa, zz)
    print(f"  s={s}: access gain on true labels {g:+.3f}")
    if best is None or abs(g - 0.05) < abs(best[1] - 0.05):
        best = (s, g, zz)
s, g, zz = best
print(f"chosen s={s}: access gain {g:+.3f}, prevalence {zz.mean():.3f} (real {rate:.3f})")
new = dict(raw); zf = np.zeros_like(raw["z"]); zf[m] = zz; new["z"] = zf
np.savez_compressed(P / f"flip/gorkha_{VARIANT}.npz", **new)
