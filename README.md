# Occupancy Modelling of Biased Landslide Inventories

Code, results and data grids for the preprint *Occupancy Modelling of Biased Landslide Inventories: When Does Separating Occurrence from Recording Improve on Neutralisation?*

Preprint: [arXiv:2611.12437](https://arxiv.org/abs/2611.12437)

**3D story:** [Seen & Unseen — Hokkaido 2018](https://rashiedomar.github.io/occupancy-landslide-inventories/), an interactive companion for non-specialists (see [The 3D story](#the-3d-story)).

## What this repository does

Landslide inventories record where landslides were seen, not where they occurred. The established remedy fits bias-describing variables, such as road distance, and neutralises them when mapping. This repository tests an alternative from ecology: a deep occupancy–detection model that treats several inventories as imperfect observers of one event.

For each cell *i* and inventory *k*:

```
z_i  ~ Bernoulli(psi_i)        psi_i = f(terrain)         did a landslide occur?
y_ik ~ Bernoulli(z_i * p_ik)   p_ik  = g_k(accessibility) did inventory k record it?
```

Only `psi` is used for mapping. The models are in [`toy_simulation/models.py`](toy_simulation/models.py).

## Main findings

- **Against standard practice:** the occupancy model reduces road bias in 24 of 25 paired draws. AUC rises by 0.08–0.14 on Gorkha, 0.03 on Haiti and 0.01–0.02 on Iburi.
- **Landslide rate:** its median predicted-to-true rate is 0.99 (range 0.44–1.89), against 0.24–0.30 for the baselines. Rescaling the baselines with the same anchor area corrects their median but leaves larger errors (0.20–4.34).
- **Calibration:** cell-level probabilities remain too extreme (calibration slope 0.59).
- **Against neutralisation:** AUC is higher on Iburi (by 0.03–0.07) and lower on Gorkha and Haiti (by 0.01–0.05).
- **When to use which:** in all eight settings with a low interaction gap, neutralisation was at least as accurate. A high gap did not guarantee an advantage.
- **Real inventories:** the three real Gorkha inventories are slope-biased, not road-biased, and the model brings no meaningful gain on them.

Most of these experiments use near-complete inventories thinned by simulated reporting bias. Report-based catalogues remain untested.

## Setup

Tested with Python 3.11 on macOS, on CPU.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Check the saved results

This step fits no models and takes a few seconds. It rebuilds the numbers behind the paper's tables and its figures from the saved results.

```bash
cd real_terrain
python rev_summary.py           # writes results_rev/summary.json
python make_paper_figures.py    # writes ../paper/latex/figures/fig_*.png
```

## Rerun the experiments

All commands run from `real_terrain/`. One model draw takes a few minutes on a CPU, and the full set takes many hours.

```bash
# Iburi
python run_iburi.py
python run_iburi_v3.py --scenarios AB
python run_iburi_v3.py --scenarios C
python run_iburi_v4.py --scenario B
python run_iburi_v4.py --scenario C
python run_steger_baseline.py
python run_extra.py --scenario D
python run_extra.py --scenario E

# Gorkha and Haiti
python run_headline.py --grid ../data/processed/gorkha_grid.npz --out results_gorkha --scenario A
python run_headline.py --grid ../data/processed/gorkha_grid.npz --out results_gorkha --scenario C
python run_headline.py --grid ../data/processed/haiti_grid.npz --out results_haiti --scenario A

# Transplant worlds: writes data/processed/flip/, then fit each world with run_headline.py
python q2_flip_worlds.py
python q2_transplant.py
python q2_transplant.py extra
python q2_proxy_world.py H1
python run_headline.py --grid ../data/processed/flip/iburi_W1.npz --out results_flip/iburi_W1 --scenario A

# Real Gorkha inventories
python q1_run_real.py
python q1_spillover.py

# Selection rule, calibration and neutralised GAM. The transplant worlds work here too.
python rev_rule_checks.py iburi A
python rev_calibration.py iburi A 0     # site, scenario, draw
python rev_gam_baseline.py iburi A
```

## Where each part of the paper comes from

| Paper section | Scripts | Saved results (`real_terrain/`) |
|---|---|---|
| Occupancy model, standard practice and neutralisation | `run_iburi*.py`, `run_steger_baseline.py`, `run_headline.py`, `rev_gam_baseline.py` | `results`, `results_v3`, `results_v4`, `results_steger`, `results_gorkha`, `results_haiti`, `results_rev` |
| Estimating landslide probability | `rev_calibration.py` | `results_rev/calib_*` |
| When the occupancy model outperforms neutralisation | `q2_*.py`, `rev_rule_checks.py`, `rev_rerun_flip.py` | `results_flip`, `results_q2`, `results_rev/rule_checks_*` |
| Prespecified test on Haiti | `q3_preregister.py`, `run_headline.py` | `results_q3/haiti_prediction.json`, `results_haiti` |
| Real, independently produced inventories | `q1_real_inventories.py`, `q1_run_real.py`, `q1_spillover.py` | `results_q1` |
| Robustness and design choices | `run_extra.py`, `make_paper_figures.py` | `results_extra`, `results_v3` |

## Reading the code

The code keeps the names used while the work was under way. Where they differ from the paper, the paper's wording is the accurate one.

- **"Steger" labels** mean the neutralisation baselines. They use bias variables as fixed effects, set to their mean when mapping. Steger et al. (2021) use a GAMM with bias variables as random effects, so these baselines are not an exact replication of their method.
- **"Pre-registered"**, as in `q3_preregister.py`, means prespecified. The Haiti prediction carries a UTC timestamp written by the script (`written_utc`). It was not lodged with a third-party registry, and this repository's commit dates are later than the test, so they do not independently confirm the order of events.
- **Older outputs:** `summary` files in the `results*` folders, and `summarize_claims.py`, reflect earlier stages of the analysis and may state claims more strongly than the paper does. For the published figures, use `results_rev/summary.json` and the paper.
- **`toy_simulation/`** is an early synthetic test and is not reported in the paper. It is kept because `models.py`, used throughout, lives there.

## The 3D story

[`docs/`](docs/) holds *Seen & Unseen*, an interactive 3D story about the 2018 Hokkaido Eastern Iburi landslides, served with GitHub Pages. It has five steps: the ground, the earthquake, what gets recorded, what a standard model learns, and how the occupancy model separates the two. You can drag to rotate, scroll to zoom and click any landslide for its details.

It shows a 24 × 24 km block with real data:

- **Terrain:** Copernicus GLO-30 elevation.
- **Landslides:** the 5,412 Zhang et al. (2019) landslides inside the block, out of 5,625 mapped.
- **Roads:** OpenStreetMap roads, motorway to tertiary.
- **Records and model maps:** the paper's simulated records and model maps from draw 0, scenario A, as in the paper's Fig. 3. Model maps appear only in the held-out test blocks.

The soil layers drawn on the block's sides are illustrative.

To run it locally, run `python3 -m http.server 8000` in `docs/` and open http://localhost:8000. It loads three.js r170 from jsDelivr and fonts from Google Fonts, so it needs an internet connection.

`docs/story-data.js` is built by `real_terrain/make_story_data.py`, which needs the raw downloads.

## Data

The processed grids are included, so the commands above run without any downloads. See [`data/README.md`](data/README.md) for sources, licences and how to rebuild them.

## Licence

The code is under the [MIT licence](LICENSE). The data keep their sources' terms, listed in [`data/README.md`](data/README.md).
