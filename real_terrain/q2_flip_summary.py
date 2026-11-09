"""Summarise the flip experiment: does our advantage over Steger follow the nonlinearity of the truth?"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
R = HERE / "results_flip"
WORLDS = [("iburi_L", "Iburi terrain,\nadditive truth"), ("iburi_N", "Iburi terrain,\nnonlinear truth"),
          ("iburi_W1", "Iburi terrain,\nreal fit: terrain"), ("iburi_W2", "Iburi terrain,\nreal fit: terrain+roads"),
          ("iburi_W3", "Iburi terrain,\nreal fit: no epicentre"), ("iburi_W1a", "Iburi terrain,\nreal fit: additive"),
          ("gorkha_H1", "Gorkha terrain,\nhidden road-tied driver"),
          ("gorkha_L", "Gorkha terrain,\nadditive truth"), ("gorkha_N", "Gorkha terrain,\nnonlinear truth")]
MODELS = [("Standard (terrain + access)", "Standard", "#9aa0a6"),
          ("Steger-MLP (access = mean)", "Steger (MLP)", "#e6ab02"),
          ("Steger-GLM (access neutralised)", "Steger (GLM)", "#a6761d"),
          ("OURS full", "Ours", "#08519c"), ("Oracle", "Oracle", "#2ca25f")]


def load(w):
    res = {}
    for f in sorted((R / w).glob("raw_results_A*.json")):
        res.update(json.loads(f.read_text()))
    return [v for _, v in sorted(res.items(), key=lambda kv: int(kv[0]))]


lines = ["| world | n | Standard | Steger-MLP | Steger-GLM | Ours | Oracle | ours − best Steger | ours beats best Steger |",
         "|---|---|---|---|---|---|---|---|---|"]
fig, ax = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
for i, (w, lab) in enumerate(WORLDS):
    rs = load(w)
    if not rs:
        continue
    auc = {k: np.array([r[k]["auc_true_landslides"] for r in rs]) for k, *_ in MODELS}
    best_steger = np.maximum(auc["Steger-MLP (access = mean)"], auc["Steger-GLM (access neutralised)"])
    diff = auc["OURS full"] - best_steger
    lines.append(f"| {w} | {len(rs)} | " + " | ".join(f"{auc[k].mean():.3f}" for k, *_ in MODELS)
                 + f" | {diff.mean():+.3f} | {int((diff > 0).sum())}/{len(rs)} |")
    for j, (k, name, c) in enumerate(MODELS):
        x = i + (j - 2) * 0.16
        ax[0].bar(x, auc[k].mean(), 0.15, color=c, label=name if i == 0 else None)
        ax[0].scatter(np.full(len(rs), x), auc[k], s=6, c="k", zorder=3)
    ax[1].bar(i, diff.mean(), 0.6, color="#08519c" if diff.mean() > 0 else "#e6ab02")
    ax[1].scatter(np.full(len(rs), i), diff, s=10, c="k", zorder=3)
for a in ax:
    a.set_xticks(range(len(WORLDS))); a.set_xticklabels([l for _, l in WORLDS], fontsize=8.5)
ax[0].set_ylim(0.6, 0.95); ax[0].set_ylabel("AUC vs true landslides"); ax[0].legend(fontsize=7.5, ncol=5, loc="upper left")
ax[1].axhline(0, c="k", lw=0.7); ax[1].set_ylabel("AUC: ours − best Steger variant")
ax[1].set_title("> 0: occupancy model wins", fontsize=9.5)
fig.suptitle("Q2 causal tests: same real terrain, roads and reporting bias; only the TRUE susceptibility is changed", fontsize=10.5)
fig.savefig(HERE / "figures/q2_flip.png", dpi=150)
txt = "\n".join(lines)
print(txt)
(R / "summary.md").write_text(txt)
