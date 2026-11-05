# Dual-branch network vs. the other models

- Cross-validation: `family_group`, identical folds for every model
- Delta columns are measured against the **best other model for that target** by `pooled_spearman`

The architecture routes the 22 tabular columns and the 256-dimensional protein embedding down separate branches, squeezes the embedding through a learned 32-dimensional bottleneck, and fuses them through residual blocks. Early stopping runs on a split carved out of the training fold, so the out-of-fold rows below were never used to pick the stopping epoch.

## stability_ddG

| model             |   pooled_rmse |   pooled_mae |   pooled_r2 |   pooled_pearson |   pooled_spearman |   delta_vs_best_baseline_rmse |   delta_vs_best_baseline_mae |   delta_vs_best_baseline_r2 |   delta_vs_best_baseline_pearson |   delta_vs_best_baseline_spearman |
|:------------------|--------------:|-------------:|------------:|-----------------:|------------------:|------------------------------:|-----------------------------:|----------------------------:|---------------------------------:|----------------------------------:|
| ridge             |         0.457 |        0.355 |       0.241 |            0.513 |             0.507 |                         0.031 |                        0.017 |                      -0.100 |                           -0.085 |                            -0.050 |
| random_forest     |         0.442 |        0.346 |       0.291 |            0.548 |             0.522 |                         0.016 |                        0.008 |                      -0.050 |                           -0.050 |                            -0.035 |
| gradient_boosting |         0.426 |        0.338 |       0.341 |            0.598 |             0.556 |                         0.000 |                        0.000 |                       0.000 |                            0.000 |                             0.000 |
| multitask_nn      |         0.448 |        0.346 |       0.271 |            0.531 |             0.551 |                         0.022 |                        0.008 |                      -0.069 |                           -0.067 |                            -0.006 |
| dual_branch_nn    |         0.495 |        0.385 |       0.110 |            0.379 |             0.390 |                         0.069 |                        0.047 |                      -0.231 |                           -0.219 |                            -0.166 |

## functional_score

| model             |   pooled_rmse |   pooled_mae |   pooled_r2 |   pooled_pearson |   pooled_spearman |   delta_vs_best_baseline_rmse |   delta_vs_best_baseline_mae |   delta_vs_best_baseline_r2 |   delta_vs_best_baseline_pearson |   delta_vs_best_baseline_spearman |
|:------------------|--------------:|-------------:|------------:|-----------------:|------------------:|------------------------------:|-----------------------------:|----------------------------:|---------------------------------:|----------------------------------:|
| ridge             |         0.079 |        0.063 |       0.529 |            0.730 |             0.708 |                         0.009 |                        0.007 |                      -0.105 |                           -0.067 |                            -0.040 |
| random_forest     |         0.073 |        0.058 |       0.592 |            0.776 |             0.721 |                         0.004 |                        0.003 |                      -0.042 |                           -0.020 |                            -0.027 |
| gradient_boosting |         0.069 |        0.055 |       0.634 |            0.796 |             0.747 |                         0.000 |                        0.000 |                       0.000 |                            0.000 |                             0.000 |
| multitask_nn      |         0.149 |        0.126 |      -0.677 |            0.222 |             0.218 |                         0.079 |                        0.070 |                      -1.311 |                           -0.574 |                            -0.529 |
| dual_branch_nn    |         0.115 |        0.091 |      -0.002 |            0.301 |             0.282 |                         0.045 |                        0.036 |                      -0.636 |                           -0.496 |                            -0.465 |

