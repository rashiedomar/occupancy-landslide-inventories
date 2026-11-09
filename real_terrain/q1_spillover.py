"""
Q1 follow-up: fix the false-positive extension.
Diagnosis: 59-84% of each real inventory's 'extra' cells lie within one cell of a reference landslide
(vs 20% for a random cell) -> mostly positional mismatch, not invented landslides.
Fix: SpilloverOccupancy - false positives arise mainly next to (predicted) landslides.
Same 5 draws, splits and anchors as q1_run_real.py; only the new variant is trained, then merged.
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
from models import MonotoneDetection, SpilloverOccupancy, mlp, neighbour_index, train  # noqa: E402
from q1_run_real import ACC_SIGNS, GRID, OBS, OUT, VIS, metrics, setup  # noqa: E402
from run_iburi import load_grid  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

torch.set_num_threads(2)


def run_one(seed, grid, obs, nbr, restarts=3, steps=3000):
    D = setup(seed, grid, obs)
    Xt, Xa, Xv = D["Xt"], D["Xa"], D["Xv"]
    Xo = np.concatenate([Xa, Xv, Xv], 1)
    signs = ACC_SIGNS + [+1] * len(VIS) + [-1] * len(VIS)
    tr_idx = torch.tensor(np.flatnonzero(D["tr"])); va_idx = torch.tensor(np.flatnonzero(D["va"]))
    T = lambda a: torch.tensor(a)
    preds, learned = [], []
    for r in range(restarts):
        torch.manual_seed(1000 * seed + r)
        mm = SpilloverOccupancy(mlp(Xt.shape[1], 1), MonotoneDetection(signs, 3), 4, T(Xt), T(Xo), T(D["Y"]),
                                T(D["M"]), T(nbr), fixed_p=[None, None, None, 0.99], fp_mask=[1, 1, 1, 0])
        train(mm, lambda: mm.loss(tr_idx), val_loss_fn=lambda: mm.loss(va_idx), steps=steps, seed=1000 * seed + r)
        preds.append(mm.predict(T(Xt[D["te"]])).numpy())
        with torch.no_grad():
            learned.append(dict(detection=[float(v) for v in mm.predict_detection(T(Xo)).numpy().mean(0)],
                                spill=[float(v) for v in torch.sigmoid(mm.a).numpy()[:3]],
                                background=[float(v) for v in torch.sigmoid(mm.b).numpy()[:3]]))
    p = np.mean(preds, 0)
    out = metrics(p, D)
    return out, learned, p


def main():
    grid = load_grid(GRID)
    obs = dict(np.load(OBS))
    nbr = neighbour_index(grid[1])
    prev = json.loads((OUT / "raw_results.json").read_text())
    for s in range(5):
        t0 = time.time()
        out, learned, p = run_one(s, grid, obs, nbr)
        prev[str(s)]["OURS + visibility + spillover false positives"] = out
        prev[str(s)]["_learned"]["OURS + visibility + spillover false positives"] = learned
        print(f"draw {s}: {time.time() - t0:.0f}s AUC {out['auc_reference']:.3f} rate {out['rate_over_reference']:.2f} "
              f"steep/gentle {out['steep_gentle_ratio']:.2f} | spill {np.round(learned[0]['spill'], 2)} "
              f"detection {np.round(learned[0]['detection'], 2)}", flush=True)
        (OUT / "raw_results.json").write_text(json.dumps(prev, indent=1, default=float))


if __name__ == "__main__":
    main()
