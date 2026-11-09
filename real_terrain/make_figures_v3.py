"""Figures for Stage 3 (run after run_iburi_v3.py)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "toy_simulation"))
from models import MonotoneDetection  # noqa: E402
from run_iburi_v3 import ACCESS, OUT, SIGNS  # noqa: E402

FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

MODELS = [("Standard (terrain + access)", "Standard\n(+road feats)", "#9aa0a6"),
          ("Standard (terrain only)", "Standard\n(roads dropped)", "#bdbdbd"),
          ("Anchor area only (terrain)", "Anchor area\nonly", "#e6ab02"),
          ("Occupancy v1 (MLP detection)", "Occupancy v1", "#c6dbef"),
          ("Occupancy + monotone", "+ monotone", "#6baed6"),
          ("Occupancy + anchor", "+ anchor", "#9ecae1"),
          ("Occupancy + monotone + anchor (FULL)", "+ monotone\n+ anchor (full)", "#08519c"),
          ("Oracle (full real inventory)", "Oracle", "#2ca25f")]
TITLES = {"A": "A: terrain-only susceptibility", "B": "B: susceptibility also sees roads",
          "C": "C: stress test (misspecified detection)"}


def load():
    res = {}
    for f in sorted(OUT.glob("raw_results_*.json")):
        res.update(json.loads(f.read_text()))
    return res


def fig_summary(res):
    mets = [("auc_true_landslides", "AUC vs REAL landslides", (0.6, 0.9)),
            ("auc_remote_quarter", "AUC in most remote quarter", (0.4, 0.9)),
            ("abs_log_ratio", "|log remote-vs-near ratio|\n(0 = unbiased; lower is better)", None)]
    scens = [s for s in "ABC" if s in res]
    fig, ax = plt.subplots(len(mets), len(scens), figsize=(4.6 * len(scens), 8.5),
                           constrained_layout=True, sharex=True, squeeze=False)
    x = np.arange(len(MODELS))
    for j, s in enumerate(scens):
        runs = res[s]
        for i, (met, lab, lim) in enumerate(mets):
            a = ax[i, j]
            vals = [np.array([r[k][met] for r in runs]) for k, _, _ in MODELS]
            a.bar(x, [v.mean() for v in vals], color=[c for *_, c in MODELS], width=0.72)
            for xi, v in zip(x, vals):
                a.scatter(np.full(len(v), xi) + np.linspace(-0.17, 0.17, len(v)), v, s=7, c="k", zorder=3)
            if lim:
                a.set_ylim(*lim)
            if j == 0:
                a.set_ylabel(lab)
            if i == 0:
                a.set_title(TITLES[s], fontsize=9.5)
            a.set_xticks(x); a.set_xticklabels([n for _, n, _ in MODELS], fontsize=6.8, rotation=90)
    fig.suptitle(f"Stage 3 — real Iburi terrain & landslides, {len(runs)} random draws per scenario "
                 "(dots), bar = mean", fontsize=10)
    fig.savefig(FIG / "v3_summary_bars.png", dpi=150)
    plt.close(fig)


def fig_detection(scen="A"):
    """Learned detection curves of the FULL model (seed 0) + learned vs true detection."""
    P = np.load(OUT / f"seed0_{scen}.npz")
    states = torch.load(OUT / f"seed0_{scen}_detection_heads.pt")
    heads = []
    for st in states:
        h = MonotoneDetection(SIGNS, 3); h.load_state_dict(st); heads.append(h)
    names = ["news catalog", "agency survey", "satellite map"]
    cols = ["#d95f02", "#7570b3", "#1b9e77"]
    labels = {"log_d_major": "distance to major road", "log_d_public": "distance to public road",
              "log_d_track": "distance to track", "built": "built-up density", "cloud": "cloud cover"}
    fig, ax = plt.subplots(1, 6, figsize=(17, 3.4), constrained_layout=True)
    grid = torch.linspace(-2.5, 2.5, 100)
    for j, f in enumerate(ACCESS):
        a = ax[j]
        X = torch.zeros(100, len(ACCESS)); X[:, j] = grid
        with torch.no_grad():
            comp = np.mean([h.component(X)[:, :, j].numpy() for h in heads], 0)   # (100, K)
        for k in range(3):
            a.plot(grid, comp[:, k] - comp[50, k], c=cols[k], label=names[k])
        a.axhline(0, c="k", lw=0.5)
        a.set_title(f"effect of {labels[f]}", fontsize=8.5); a.set_xlabel("standardised value")
        if j == 0:
            a.set_ylabel("change in detection log-odds"); a.legend(fontsize=7)
    a = ax[5]
    ph, pt, cov = P["p_hat"], P["p_true"], P["cover"].astype(bool)
    rng = np.random.default_rng(0)
    for k in range(3):
        idx = np.flatnonzero(cov[:, k]); idx = rng.choice(idx, min(1500, len(idx)), replace=False)
        a.scatter(pt[idx, k], ph[idx, k], s=2, c=cols[k], alpha=0.5, label=names[k])
    a.plot([0, 1], [0, 1], "k--", lw=0.7); a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.set_xlabel("TRUE detection prob."); a.set_ylabel("learned"); a.set_title("learned vs true detection", fontsize=8.5)
    fig.suptitle(f"What the full model learned about each observer (scenario {scen}, draw 0; "
                 "curves constrained to the known direction)", fontsize=10)
    fig.savefig(FIG / f"v3_detection_curves_{scen}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    res = load()
    fig_summary(res)
    for s in "AC":
        if (OUT / f"seed0_{s}_detection_heads.pt").exists():
            fig_detection(s)
    print("figures in", FIG)
