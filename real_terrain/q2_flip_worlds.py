"""
Q2 causal test: dial the nonlinearity of the TRUE susceptibility on BOTH real terrains.

For each terrain (Iburi, Gorkha), two synthetic truths on the real terrain, real roads, same prevalence:
  L (additive-linear): logit = GLM fitted to the real inventory (terrain features only)
  N (nonlinear)       : logit = same GLM logit + h(terrain), where h is a fixed smooth random network
                        with interactions, scaled so the GBM-vs-GLM AUC gap on the truth matches Iburi's
                        real gap (~ +0.09).
Truth z ~ Bernoulli(psi) once per world (fixed seed). Then the usual biased reporting is simulated and the
headline models are compared (run_headline.py on each world grid).

Prediction if nonlinearity is the cause: Steger catches up in L worlds and falls behind in N worlds, on
BOTH terrains -- i.e. the Iburi/Gorkha pattern can be flipped.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run_iburi import _z, load_grid  # noqa: E402

GRIDS = {"iburi": HERE.parent / "data/processed/iburi_grid.npz",
         "gorkha": HERE.parent / "data/processed/gorkha_grid.npz"}
OUT = HERE.parent / "data/processed/flip"; OUT.mkdir(exist_ok=True)
TARGET_GAP = 0.09


def sig(x):
    return 1 / (1 + np.exp(-x))


def random_net(X, seed, width=32):
    """Fixed smooth random function with interactions (2 tanh layers), standardised output."""
    r = np.random.default_rng(seed)
    W1 = r.normal(0, 1.2 / np.sqrt(X.shape[1]), (X.shape[1], width)); b1 = r.normal(0, 0.5, width)
    W2 = r.normal(0, 1.2 / np.sqrt(width), (width, width)); b2 = r.normal(0, 0.5, width)
    w3 = r.normal(0, 1 / np.sqrt(width), width)
    h = np.tanh(np.tanh(X @ W1 + b1) @ W2 + b2) @ w3
    return (h - h.mean()) / h.std()


def interactions(X, names):
    """Pure interaction / threshold structure with ~zero linear main effect (invisible to a GLM):
    'steep slopes fail where the ground is concave', 'relief matters on one aspect', and a slope band."""
    c = {n: X[:, i] for i, n in enumerate(names)}
    h = (c["slope_mean"] * -c["curvature"] + c["relief"] * c["aspect_cos"] + c["tpi_150m"] * c["elev"]
         + 1.5 * (np.exp(-(c["slope_mean"] - 0.8) ** 2) - 0.6))
    h = h - np.polyval(np.polyfit(np.arange(len(h)), h, 0), 0)          # centre
    # remove the part a GLM could capture (project out linear terms)
    A = np.c_[np.ones(len(h)), X]
    h = h - A @ np.linalg.lstsq(A, h, rcond=None)[0]
    return h / h.std()


def calibrate(logit, rate):
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if sig(logit + mid).mean() > rate else (mid, hi)
    return logit + mid


def gap(X, z, seed=0):
    r = np.random.default_rng(seed); idx = r.permutation(len(z)); k = int(0.7 * len(z))
    tr, te = idx[:k], idx[k:]
    glm = LogisticRegression(max_iter=2000).fit(X[tr], z[tr])
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X[tr], z[tr])
    a = lambda m: roc_auc_score(z[te], m.predict_proba(X[te])[:, 1])
    return a(gbm) - a(glm), a(glm), a(gbm)


def main():
    for name, path in GRIDS.items():
        d, m, terrain, dist = load_grid(path)
        raw = dict(np.load(path, allow_pickle=True))
        names = sorted(terrain)
        X = np.stack([_z(terrain[k], m)[m] for k in names], -1)
        z_real = d["z"][m].astype(int)
        rate = z_real.mean()
        glm_logit = LogisticRegression(max_iter=3000).fit(X, z_real).decision_function(X)
        print(f"{name}: real gap GBM-GLM {gap(X, z_real)[0]:+.3f}, prevalence {rate:.3f}")
        h = interactions(X, names)
        for world, scale in [("L", 0.0), ("N", None)]:
            if scale is None:                                   # search scale to hit the target gap
                best = None
                for s in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
                    lg = calibrate(glm_logit + s * h, rate)
                    zz = (np.random.default_rng(123).random(len(lg)) < sig(lg)).astype(int)
                    g = gap(X, zz)[0]
                    if best is None or abs(g - TARGET_GAP) < abs(best[1] - TARGET_GAP):
                        best = (s, g)
                scale = best[0]
            lg = calibrate(glm_logit + scale * h, rate)
            zz = (np.random.default_rng(123).random(len(lg)) < sig(lg)).astype(np.int8)
            g, aglm, agbm = gap(X, zz)
            print(f"  world {world}: scale {scale}, truth gap GBM-GLM {g:+.3f} (GLM {aglm:.3f}, GBM {agbm:.3f}), "
                  f"prevalence {zz.mean():.3f}")
            new = dict(raw)
            zfull = np.zeros_like(raw["z"]); zfull[m] = zz
            new["z"] = zfull
            np.savez_compressed(OUT / f"{name}_{world}.npz", **new)


if __name__ == "__main__":
    main()
