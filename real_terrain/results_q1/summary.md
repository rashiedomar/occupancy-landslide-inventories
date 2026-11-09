| model (5 draws) | AUC vs reference | AUC vs consensus | gentle-slope ¼ AUC | steep/gentle ratio (truth ≈ oracle) | predicted ÷ true rate |
|---|---|---|---|---|---|
| Standard (terrain + access) | 0.830 ± 0.013 | 0.867 | 0.833 | 1.61 | 1.11 |
| Standard (terrain only) | 0.836 ± 0.014 | 0.872 | 0.841 | 1.56 | 1.13 |
| Steger-MLP (access = mean) | 0.829 ± 0.013 | 0.863 | 0.843 | 1.30 | 1.15 |
| Steger-GLM (access neutralised) | 0.825 ± 0.014 | 0.858 | 0.853 | 1.29 | 1.05 |
| OURS (access detection) | 0.837 ± 0.014 | 0.873 | 0.839 | 1.66 | 1.38 |
| OURS + visibility detection | 0.835 ± 0.015 | 0.872 | 0.839 | 1.37 | 1.35 |
| OURS + visibility + false positives | 0.837 ± 0.014 | 0.873 | 0.848 | 2.50 | 1.15 |
| Oracle (reference inventory) | 0.852 ± 0.014 | 0.867 | 0.863 | 1.24 | 0.99 |