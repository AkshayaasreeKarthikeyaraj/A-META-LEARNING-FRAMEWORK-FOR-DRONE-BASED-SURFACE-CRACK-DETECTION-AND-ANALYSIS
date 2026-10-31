# Final metrics report

## Table 1 - Metrics of the three trained models (test split, %)

| Dataset | Test images | Adapt steps | Threshold | Dice/F1 | IoU | Precision | Recall | Accuracy | Specificity | ROC-AUC | Mean image Dice | Mean image IoU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Volker | 149 | 5 | 0.3 | 69.21 | 52.92 | 64.13 | 75.18 | 98.66 | 99.14 | 98.33 | 69.14 | 53.7 |
| Crack500 | 505 | 5 | 0.5 | 70.28 | 54.18 | 66.29 | 74.77 | 96.29 | 97.63 | 96.2 | 65.37 | 50.86 |
| OUC | 195 | 3 | 0.6 | 71.21 | 55.29 | 67.52 | 75.31 | 98.9 | 99.33 | 98.54 | 70.86 | 56.31 |

Dice/F1 and IoU are computed over all test pixels together; 'Mean image Dice/IoU' average the per-image scores over test images that contain a crack.

## Table 2 - Mean ± std over 3 runs (3 different random support sets, %)

| Dataset | Adapt steps | Runs | Dice/F1 | Mean image Dice | IoU | Precision | Recall | Accuracy | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Volker | 5 | 3 | 68.53 ± 0.83 | 68.38 ± 0.82 | 52.14 ± 0.96 | 61.74 ± 1.68 | 77.11 ± 2.24 | 98.58 ± 0.06 | 98.39 ± 0.09 |
| Crack500 | 5 | 3 | 69.16 ± 1.95 | 64.95 ± 1.08 | 52.89 ± 2.26 | 69.96 ± 2.62 | 68.76 ± 5.20 | 96.42 ± 0.16 | 95.89 ± 0.60 |
| OUC | 3 | 3 | 70.89 ± 0.26 | 70.47 ± 0.45 | 54.91 ± 0.31 | 65.91 ± 2.16 | 76.86 ± 2.55 | 98.86 ± 0.05 | 98.59 ± 0.10 |

## Table 3 - Dice (%) compared with the base paper (Xu et al., Table I)

| Dataset | Paper MAML-SegNet | Paper EFF-MAML-SegNet | Paper best baseline | Ours (pooled Dice) | Ours (per-image Dice) | Δ vs paper MAML-SegNet (pooled) | Δ vs paper MAML-SegNet (per-image) |
|---|---|---|---|---|---|---|---|
| Volker | 67.37 ± 0.35 | 66.83 ± 0.35 | 66.11 ± 0.60 (LETNet) | 68.53 ± 0.83 | 68.38 ± 0.82 | +1.16 | +1.01 |
| Crack500 | 61.26 ± 0.08 | 55.42 ± 1.64 | 48.07 ± 2.47 (LCNet) | 69.16 ± 1.95 | 64.95 ± 1.08 | +7.90 | +3.69 |
| OUC | 56.35 ± 1.40 | 48.03 ± 1.29 | 49.23 ± 4.91 (ERFNet) | 70.89 ± 0.26 | 70.47 ± 0.45 | +14.54 | +14.12 |

Δ is in percentage points (ours minus the paper's MAML-SegNet mean).

## Table 4 - Experimental setup differences

| Dataset | Paper: train images | Paper: val images | Paper: test images | Ours: train images | Ours: test images | Train-set size ratio (ours/paper) |
|---|---|---|---|---|---|---|
| Volker | 50 | 30 | 909 | 693 | 149 | 13.9x |
| Crack500 | 50 | 30 | 2207 | 2354 | 505 | 47.1x |
| OUC | 50 | 30 | 1211 | 902 | 195 | 18.0x |

The paper trains on only 50 images per data set and tests on the rest; this project trains on the full 70% training split. The test sets are therefore different, and the comparison is indicative, not like-for-like. The paper's own Table V shows its Dice rising with more training images (3 x 100 images: Volker 70.08, Crack500 64.01, OUC 59.93).

## Table 5 - Effect of adaptation (paper: with vs without MAML; ours: 0 steps vs adapted)

| Dataset | Paper SEDNet without MAML | Paper SEDNet with MAML | Paper gain (pp) | Ours 0 steps (no adaptation) | Ours adapted | Our gain (pp) |
|---|---|---|---|---|---|---|
| Volker | 66.40 | 67.86 | +1.46 | 61.84 ± 0.00 | 68.53 ± 0.83 | +6.70 |
| Crack500 | 39.06 | 61.32 | +22.26 | 25.74 ± 0.00 | 69.16 ± 1.95 | +43.42 |
| OUC | 46.03 | 56.23 | +10.20 | 70.45 ± 0.00 | 70.89 ± 0.26 | +0.44 |

These are not the same experiment: the paper trains a separate non-MAML network; here the same MAML-trained network is evaluated before and after test-time adaptation.

## Table 6 - Dice (%) versus number of adaptation steps (mean ± std, 3 runs)

| Dataset | 0 steps | 1 steps | 2 steps | 3 steps | 5 steps |
|---|---|---|---|---|---|
| Volker | 61.84 ± 0.00 | 63.98 ± 0.36 | 65.58 ± 0.59 | 66.80 ± 0.73 | 68.53 ± 0.83 |
| Crack500 | 25.74 ± 0.00 | 44.82 ± 4.61 | 57.77 ± 5.18 | 64.08 ± 3.95 | 69.16 ± 1.95 |
| OUC | 70.45 ± 0.00 | 70.80 ± 0.19 | 70.92 ± 0.29 | 70.89 ± 0.26 | 70.80 ± 0.13 |

## Table 7 - Model size and speed

| Model | Params (M) | GPU FPS (Colab) | CPU FPS (Colab) |
|---|---|---|---|
| Ours (Volker) | 3.069 | 94.9 | 3.45 |
| Ours (Crack500) | 3.069 | 95.1 | 2.54 |
| Ours (OUC) | 3.069 | 97.3 | 3.51 |
| Paper SEDNet (MAML-SegNet) | 1.17 | - | 4.95 (Jetson Xavier NX) |
| Paper EFF-SEDNet | 0.98 | - | 25.54 (Jetson Xavier NX) |

Speeds were measured on different hardware (Colab vs Jetson Xavier NX) and are not directly comparable.
