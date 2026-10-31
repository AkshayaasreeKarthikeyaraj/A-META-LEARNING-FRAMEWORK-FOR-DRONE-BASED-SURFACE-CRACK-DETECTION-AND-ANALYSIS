# Ablation study report

## Table 1 - Ablation: Dice (%), mean ± std over 3 support sets, after test-time adaptation

| ID | Variant | MAML | Refinement | Lref | Volker | Crack500 | OUC | Average |
|---|---|---|---|---|---|---|---|---|
| A1 | MAML only | yes | no | no | 68.96 ± 1.06 | 65.95 ± 2.78 | 70.89 ± 0.34 | 68.60 |
| A2 | Refinement only (supervised) | no | yes | no | 67.98 ± 0.22 | 56.31 ± 2.69 | 65.63 ± 0.28 | 63.31 |
| A3 | Refinement + Lref (supervised) | no | yes | 1 | 68.43 ± 0.92 | 63.62 ± 5.12 | 70.01 ± 1.12 | 67.35 |
| A4 | MAML + refinement | yes | yes | no | 68.18 ± 0.66 | 65.41 ± 5.30 | 70.47 ± 0.53 | 68.02 |
| A5 | MAML + refinement + Lref (full) | yes | yes | 1 | 68.53 ± 0.83 | 69.16 ± 1.95 | 70.89 ± 0.26 | 69.53 |

Every variant is adapted on the same kind of support set at test time (5 steps for Volker and Crack500, 3 for OUC), so the comparison between MAML and supervised training is like-for-like.

## Table 1b - Dice (%) with NO test-time adaptation (all variants)

| ID | Variant | MAML | Refinement | Lref | Volker | Crack500 | OUC | Average |
|---|---|---|---|---|---|---|---|---|
| A1 | MAML only | yes | no | no | 62.40 ± 0.00 | 29.48 ± 0.00 | 70.93 ± 0.00 | 54.27 |
| A2 | Refinement only (supervised) | no | yes | no | 66.63 ± 0.00 | 47.28 ± 0.00 | 65.90 ± 0.00 | 59.94 |
| A3 | Refinement + Lref (supervised) | no | yes | 1 | 63.31 ± 0.00 | 44.99 ± 0.00 | 71.29 ± 0.00 | 59.86 |
| A4 | MAML + refinement | yes | yes | no | 62.26 ± 0.00 | 30.29 ± 0.00 | 70.68 ± 0.00 | 54.41 |
| A5 | MAML + refinement + Lref (full) | yes | yes | 1 | 61.99 ± 0.00 | 26.08 ± 0.00 | 70.85 ± 0.00 | 52.97 |

## Table 2 - All metrics per variant and data set (%)

| ID | Dataset | Threshold | Dice/F1 | IoU | Precision | Recall | Mean image Dice | Coarse-loc Dice | Refined-loc Dice | Params (M) |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | Volker | 0.5 | 68.96 ± 1.06 | 52.64 ± 1.22 | 64.39 ± 1.49 | 74.26 ± 0.82 | 68.96 ± 1.06 | 63.95 | 63.95 | 2.976 |
| A1 | Crack500 | 0.3 | 65.95 ± 2.78 | 49.26 ± 3.07 | 66.09 ± 4.61 | 66.06 ± 3.49 | 62.13 ± 1.74 | 63.35 | 63.35 | 2.976 |
| A1 | OUC | 0.6 | 70.89 ± 0.34 | 54.91 ± 0.41 | 66.33 ± 2.72 | 76.39 ± 3.19 | 70.68 ± 0.41 | 58.03 | 58.03 | 2.976 |
| A2 | Volker | 0.8 | 67.98 ± 0.22 | 51.49 ± 0.26 | 59.01 ± 0.39 | 80.19 ± 0.91 | 67.66 ± 0.18 | 58.87 | 7.16 | 3.069 |
| A2 | Crack500 | 0.6 | 56.31 ± 2.69 | 39.23 ± 2.63 | 56.96 ± 1.67 | 55.82 ± 4.51 | 53.44 ± 1.93 | 51.34 | 7.24 | 3.069 |
| A2 | OUC | 0.8 | 65.63 ± 0.28 | 48.84 ± 0.31 | 59.26 ± 0.95 | 73.57 ± 1.31 | 64.81 ± 0.26 | 46.65 | 6.06 | 3.069 |
| A3 | Volker | 0.7 | 68.43 ± 0.92 | 52.02 ± 1.05 | 63.75 ± 0.60 | 73.86 ± 1.54 | 68.47 ± 0.94 | 59.05 | 60.04 | 3.069 |
| A3 | Crack500 | 0.5 | 63.62 ± 5.12 | 46.85 ± 5.37 | 69.90 ± 3.64 | 59.49 ± 9.67 | 59.86 ± 3.97 | 63.13 | 59.69 | 3.069 |
| A3 | OUC | 0.6 | 70.01 ± 1.12 | 53.87 ± 1.32 | 63.46 ± 2.38 | 78.18 ± 1.08 | 69.69 ± 1.04 | 56.59 | 57.24 | 3.069 |
| A4 | Volker | 0.5 | 68.18 ± 0.66 | 51.72 ± 0.76 | 62.65 ± 0.17 | 74.79 ± 1.42 | 68.19 ± 0.73 | 61.85 | 0.88 | 3.069 |
| A4 | Crack500 | 0.5 | 65.41 ± 5.30 | 48.82 ± 5.70 | 73.47 ± 2.03 | 59.63 ± 8.72 | 60.54 ± 3.99 | 64.63 | 8.96 | 3.069 |
| A4 | OUC | 0.6 | 70.47 ± 0.53 | 54.41 ± 0.63 | 63.90 ± 2.09 | 78.72 ± 2.14 | 70.07 ± 0.56 | 57.11 | 0.68 | 3.069 |
| A5 | Volker | 0.3 | 68.53 ± 0.83 | 52.14 ± 0.96 | 61.74 ± 1.68 | 77.11 ± 2.24 | 68.38 ± 0.82 | 63.45 | 62.77 | 3.069 |
| A5 | Crack500 | 0.5 | 69.16 ± 1.95 | 52.89 ± 2.25 | 69.96 ± 2.62 | 68.75 ± 5.19 | 64.96 ± 1.07 | 67.91 | 68.25 | 3.069 |
| A5 | OUC | 0.6 | 70.89 ± 0.26 | 54.91 ± 0.31 | 65.91 ± 2.16 | 76.86 ± 2.55 | 70.47 ± 0.45 | 58.95 | 58.92 | 3.069 |

Coarse-loc / Refined-loc Dice: Dice of the localization map (threshold 0.5) against the crack mask. Without the refinement module (A1) the refined map is the coarse map.

## Table 3 - Contribution of each part (change in Dice, percentage points)

| Effect of | Comparison | Volker | Crack500 | OUC | Average |
|---|---|---|---|---|---|
| Refinement module (under MAML) | A4 − A1 | -0.79 | -0.54 | -0.41 | -0.58 |
| Refinement loss Lref (under MAML) | A5 − A4 | +0.36 | +3.75 | +0.42 | +1.51 |
| Refinement loss Lref (supervised) | A3 − A2 | +0.45 | +7.32 | +4.38 | +4.05 |
| MAML training, no Lref | A4 − A2 | +0.19 | +9.10 | +4.85 | +4.71 |
| MAML training, with Lref | A5 − A3 | +0.10 | +5.53 | +0.88 | +2.17 |
| Full method vs refinement-only supervised | A5 − A2 | +0.55 | +12.85 | +5.26 | +6.22 |

Each row compares two variants that differ in exactly one thing. Differences below about 1 point may be within training noise (one training run per variant).

## Table 4 - MAML system vs a plainly trained model (Dice change, points)

| Comparison | Volker | Crack500 | OUC | Average |
|---|---|---|---|---|
| A4 (MAML, adapted) vs A2 (supervised, NO adaptation) | +1.54 | +18.13 | +4.57 | +8.08 |
| A5 (MAML, adapted) vs A3 (supervised, NO adaptation) | +5.22 | +24.17 | -0.40 | +9.66 |
