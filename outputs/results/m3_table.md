# Milestone 3: U-Net (ResNet-34) Results

Pooled water IoU (%) on the test split of each regime, mean ± sample std over seeds (42, 1337, 2024). Model selection uses the val split only; each seed's best checkpoint is evaluated once. ECE uses 15 equal-width probability bins over valid pixels.

## Main results (BCE + Dice)

| Modality | Official test IoU | Bolivia IoU (official model) | Held-out events IoU (LOEO) | LOEO: India | LOEO: Bolivia | Official F1 | Official ECE | Seeds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SAR (S1, 2 ch)** | 67.3 ± 1.2 | 64.8 ± 4.0 | 53.1 ± 2.8 | 51.7 ± 2.1 | 58.6 ± 12.4 | 80.5 ± 0.9 | 2.5 ± 0.4 | 3 / 3 |
| **Optical (S2, 13 ch)** | 82.5 ± 0.3 | 80.5 ± 0.5 | 73.7 ± 0.5 | 72.2 ± 0.6 | 79.7 ± 0.1 | 90.4 ± 0.2 | 1.0 ± 0.2 | 3 / 3 |
| **Early fusion (15 ch)** | 82.7 ± 1.2 | 79.3 ± 1.4 | 71.8 ± 1.9 | 70.7 ± 2.1 | 76.4 ± 2.4 | 90.5 ± 0.7 | 1.1 ± 0.2 | 3 / 3 |

## Paired modality comparisons (test split, seed-summed per-chip counts)

| Comparison | Δ pooled IoU (pp) | 95% CI (paired chip bootstrap) | p (two-sided) | Chips |
| :--- | :---: | :---: | :---: | :---: |
| official: fusion - s2 | +0.2 | [-1.3, +1.6] | 0.763 | 90 |
| official: fusion - s1 | +15.4 | [+10.0, +22.2] | <0.001 | 90 |
| official: s2 - s1 | +15.2 | [+9.7, +22.1] | <0.001 | 90 |
| loeo: fusion - s2 | -1.9 | [-2.8, -1.1] | <0.001 | 83 |
| loeo: fusion - s1 | +18.7 | [+14.4, +24.9] | <0.001 | 83 |
| loeo: s2 - s1 | +20.5 | [+16.2, +26.8] | <0.001 | 83 |

## Loss ablation (official regime, seed 42)

| Modality | BCE + Dice IoU (%) | Dice-only IoU (%) | BCE + Dice ECE (%) | Dice-only ECE (%) |
| :--- | :---: | :---: | :---: | :---: |
| SAR (S1, 2 ch) | 67.2 | 67.7 | 2.1 | 3.6 |
| Optical (S2, 13 ch) | 82.8 | 80.8 | 1.3 | 1.3 |
| Early fusion (15 ch) | 82.4 | 83.1 | 1.3 | 1.4 |

## Training cost

| Config | Params | Best epoch | Train minutes |
| :--- | :---: | :---: | :---: |
| `loeo_fusion_bce_dice` | 24.47M | 30 | 9.0 |
| `loeo_s1_bce_dice` | 24.43M | 24 | 6.4 |
| `loeo_s2_bce_dice` | 24.47M | 29 | 8.6 |
| `official_fusion_bce_dice` | 24.47M | 38 | 9.1 |
| `official_fusion_dice_only` | 24.47M | 49 | 9.5 |
| `official_s1_bce_dice` | 24.43M | 35 | 6.7 |
| `official_s1_dice_only` | 24.43M | 50 | 6.3 |
| `official_s2_bce_dice` | 24.47M | 37 | 8.3 |
| `official_s2_dice_only` | 24.47M | 30 | 7.1 |
