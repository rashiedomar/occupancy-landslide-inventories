"""Re-run one draw of a transplant world with run_headline.run_one. Usage: python rev_rerun_flip.py <world> <seed>"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "toy_simulation"))
sys.path.insert(0, str(HERE))
from rev_rule_checks import OUT, SITES, world_truth  # noqa: E402
from run_headline import run_one  # noqa: E402
from run_iburi import load_grid  # noqa: E402

world, seed = sys.argv[1], int(sys.argv[2])
torch.set_num_threads(2)
grid = load_grid(SITES[world.split("_")[0]])
d, m, terrain, dist = grid
zf = np.zeros_like(d["z"]); zf[m] = world_truth(world, grid)
raw = {k: d[k] for k in d.files}; raw["z"] = zf
res = run_one(seed, "A", (raw, m, terrain, dist), skip_roadcut=True)
(OUT / f"flip_rerun_{world}_{seed}.json").write_text(json.dumps(res, indent=1, default=float))
print(world, seed, {k: round(v["auc_true_landslides"], 3) for k, v in res.items() if not k.startswith("_")}, flush=True)
