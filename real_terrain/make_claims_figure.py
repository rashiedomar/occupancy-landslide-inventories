"""One figure for the headline claims: standard vs Steger vs ours vs oracle, per setting."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_claims import collect

FIG = Path(__file__).parent / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

sets = collect()
settings = [k for k in ["Iburi-A", "Iburi-C", "Gorkha-A", "Gorkha-C"] if k in sets]
models = [("standard", "Standard practice", "#9aa0a6"), ("steger", "Steger (MLP)", "#e6ab02"),
          ("steger_glm", "Steger (GLM, original)", "#a6761d"),
          ("ours", "Ours (occupancy)", "#08519c"), ("oracle", "Oracle (full inventory)", "#2ca25f")]
mets = [("auc_true_landslides", "AUC vs real landslides", (0.6, 0.9)),
        ("auc_remote_quarter", "AUC in most remote quarter", (0.4, 0.9)),
        ("abs_log_ratio", "Road bias |log ratio|  (lower = better)", None),
        ("abs_log_rate", "Calibration |log(predicted ÷ true rate)|\n(lower = better)", None)]
fig, ax = plt.subplots(1, 4, figsize=(19, 4.4), constrained_layout=True)
w = 0.16
for i, (met, lab, lim) in enumerate(mets):
    a = ax[i]
    for j, (key, name, c) in enumerate(models):
        means, xs = [], []
        for k, sname in enumerate(settings):
            v = np.array([r[met] for r in sets[sname][key]])
            if key not in sets[sname]:
                continue
            x = k + (j - (len(models) - 1) / 2) * w
            a.bar(x, v.mean(), w * 0.92, color=c, label=name if k == 0 else None)
            a.scatter(np.full(len(v), x) + np.linspace(-0.06, 0.06, len(v)), v, s=5, c="k", zorder=3)
    a.set_xticks(range(len(settings)))
    a.set_xticklabels([s.replace("-A", "\n(reporting bias)").replace("-C", "\n(stress test)") for s in settings])
    a.set_title(lab, fontsize=10)
    if lim:
        a.set_ylim(*lim)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Two earthquakes, real terrain and landslides, simulated biased records — 5 draws each (dots)",
             fontsize=10.5)
fig.savefig(FIG / "claims_summary.png", dpi=150, bbox_inches="tight")
print("saved", FIG / "claims_summary.png", "settings:", settings)
