# Milestone 2: Baseline Benchmark Results (Non-Learned)

Thresholds are tuned per split regime on that regime's **train split only** (maximising pooled water IoU). Otsu is scene-adaptive and untuned.

| Regime | NDWI τ | MNDWI τ | SAR VV τ (dB) |
| :--- | :---: | :---: | :---: |
| `official` | -0.06 | 0.15 | -16.8 |
| `loeo` | -0.05 | 0.14 | -16.9 |

**Water IoU** is pooled over all valid pixels of the split; **Chip-mean IoU** averages per-chip IoU (chips with no water and no predicted water count as 1.0). 95% CIs come from 2,000 chip-level bootstrap resamples of the same aggregate.

| Baseline | Modality | Eval split | Water IoU (%) | 95% CI | Chip-mean IoU (%) | 95% CI | F1 (%) | Precision (%) | Recall (%) | Chips |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NDWI** | Optical (S2) | `official_test` | **76.46** | [67.8, 83.7] | 56.76 | [50.2, 63.5] | 86.66 | 83.95 | 89.55 | 90 |
| **MNDWI** | Optical (S2) | `official_test` | **77.32** | [68.4, 84.0] | 53.52 | [47.0, 60.4] | 87.21 | 86.74 | 87.69 | 90 |
| **SAR_VV_Threshold** | SAR (S1) | `official_test` | **45.58** | [32.0, 57.3] | 27.11 | [21.6, 33.2] | 62.62 | 77.44 | 52.56 | 90 |
| **Otsu_SAR_Adaptive** | SAR (S1) | `official_test` | **21.22** | [14.7, 28.0] | 21.95 | [16.2, 28.1] | 35.00 | 23.20 | 71.25 | 90 |
| **NDWI** | Optical (S2) | `loeo_held_out` | **69.54** | [63.0, 74.8] | 48.09 | [41.9, 54.2] | 82.03 | 82.78 | 81.29 | 83 |
| **MNDWI** | Optical (S2) | `loeo_held_out` | **64.17** | [55.6, 71.3] | 42.10 | [36.2, 48.0] | 78.17 | 71.20 | 86.66 | 83 |
| **SAR_VV_Threshold** | SAR (S1) | `loeo_held_out` | **46.09** | [34.6, 55.6] | 23.52 | [18.3, 28.8] | 63.10 | 88.08 | 49.16 | 83 |
| **Otsu_SAR_Adaptive** | SAR (S1) | `loeo_held_out` | **24.39** | [17.7, 31.4] | 25.46 | [19.6, 31.5] | 39.22 | 26.03 | 79.48 | 83 |
