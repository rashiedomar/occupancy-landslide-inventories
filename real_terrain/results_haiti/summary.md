# Haiti 2010 — pre-registered out-of-sample test (scenario A, 5 draws, 210 m cells)

Prediction written first (`results_q3/haiti_prediction.json`): interaction gap of the true pattern = +0.002
(< 0.015) -> **"Steger as good or better"**.

| model | AUC vs truth | AUC remote quarter | predicted ÷ true rate | road bias \|log\| |
|---|---|---|---|---|
| Standard (terrain + access) | 0.739 | 0.768 | 0.24 | 1.70 |
| Steger-MLP | 0.784 | 0.791 | 0.19 | 0.65 |
| Steger-GLM | 0.783 | 0.803 | 0.29 | 0.39 |
| **Ours** | 0.773 | 0.767 | **0.97** | **0.28** |
| Oracle | 0.802 | 0.819 | 0.99 | 0.46 |

Ours − best Steger AUC: −0.032, −0.034, −0.016, +0.012, −0.014 (mean −0.017; ours wins 1/5) -> **prediction confirmed**.
Ours − standard: +0.034 (5/5). Ours is the only non-oracle model with a correct overall rate (0.97 vs 0.19–0.29)
and has the lowest road bias.
Truth: Harp et al. (2016) inventory; grid 63,995 cells, 10.4% landslide; EPSG:32618; window 18.18–18.50N, 72.98–72.17W.
