# Evaluation summary

- Reported cross-validation strategy: `family_group`
- Selection metric: `pooled_spearman` (pooled over all out-of-fold predictions)

## Selected models

| target           | model             |   n_samples |   pooled_rmse |   pooled_mae |   pooled_r2 |   pooled_pearson |   pooled_spearman |
|:-----------------|:------------------|------------:|--------------:|-------------:|------------:|-----------------:|------------------:|
| stability_ddG    | gradient_boosting |         840 |         0.426 |        0.338 |       0.341 |            0.598 |             0.556 |
| functional_score | gradient_boosting |         840 |         0.069 |        0.055 |       0.634 |            0.796 |             0.747 |

## All models, all strategies

| strategy      | model             | target           |   n_samples |   pooled_rmse |   pooled_mae |   pooled_r2 |   pooled_pearson |   pooled_spearman |   fold_mean_rmse |   fold_mean_mae |   fold_mean_r2 |   fold_mean_pearson |   fold_mean_spearman |   fold_std_rmse |   fold_std_mae |   fold_std_r2 |   fold_std_pearson |   fold_std_spearman |
|:--------------|:------------------|:-----------------|------------:|--------------:|-------------:|------------:|-----------------:|------------------:|-----------------:|----------------:|---------------:|--------------------:|---------------------:|----------------:|---------------:|--------------:|-------------------:|--------------------:|
| random        | ridge             | stability_ddG    |         840 |         0.406 |        0.315 |       0.401 |            0.634 |             0.644 |            0.405 |           0.315 |          0.397 |               0.634 |                0.641 |           0.031 |          0.017 |         0.069 |              0.055 |               0.042 |
| random        | ridge             | functional_score |         840 |         0.071 |        0.057 |       0.618 |            0.787 |             0.769 |            0.071 |           0.057 |          0.615 |               0.788 |                0.763 |           0.002 |          0.002 |         0.025 |              0.017 |               0.024 |
| random        | random_forest     | stability_ddG    |         840 |         0.409 |        0.318 |       0.392 |            0.635 |             0.627 |            0.408 |           0.318 |          0.389 |               0.636 |                0.626 |           0.034 |          0.021 |         0.063 |              0.063 |               0.065 |
| random        | random_forest     | functional_score |         840 |         0.067 |        0.053 |       0.664 |            0.826 |             0.792 |            0.067 |           0.053 |          0.662 |               0.830 |                0.794 |           0.004 |          0.003 |         0.022 |              0.014 |               0.007 |
| random        | gradient_boosting | stability_ddG    |         840 |         0.390 |        0.311 |       0.448 |            0.669 |             0.650 |            0.389 |           0.311 |          0.444 |               0.670 |                0.649 |           0.030 |          0.023 |         0.057 |              0.045 |               0.056 |
| random        | gradient_boosting | functional_score |         840 |         0.060 |        0.047 |       0.724 |            0.851 |             0.817 |            0.060 |           0.047 |          0.721 |               0.854 |                0.817 |           0.002 |          0.002 |         0.020 |              0.011 |               0.011 |
| protein_group | ridge             | stability_ddG    |         840 |         0.411 |        0.314 |       0.387 |            0.624 |             0.637 |            0.411 |           0.314 |          0.367 |               0.615 |                0.624 |           0.015 |          0.016 |         0.044 |              0.032 |               0.030 |
| protein_group | ridge             | functional_score |         840 |         0.072 |        0.059 |       0.602 |            0.776 |             0.756 |            0.072 |           0.059 |          0.591 |               0.773 |                0.750 |           0.003 |          0.002 |         0.064 |              0.037 |               0.035 |
| protein_group | random_forest     | stability_ddG    |         840 |         0.411 |        0.320 |       0.386 |            0.634 |             0.625 |            0.411 |           0.320 |          0.368 |               0.632 |                0.615 |           0.023 |          0.017 |         0.042 |              0.042 |               0.039 |
| protein_group | random_forest     | functional_score |         840 |         0.071 |        0.057 |       0.619 |            0.795 |             0.746 |            0.071 |           0.057 |          0.613 |               0.796 |                0.750 |           0.003 |          0.003 |         0.018 |              0.007 |               0.025 |
| protein_group | gradient_boosting | stability_ddG    |         840 |         0.387 |        0.308 |       0.454 |            0.674 |             0.656 |            0.387 |           0.308 |          0.437 |               0.670 |                0.645 |           0.007 |          0.006 |         0.061 |              0.054 |               0.037 |
| protein_group | gradient_boosting | functional_score |         840 |         0.066 |        0.053 |       0.670 |            0.819 |             0.771 |            0.066 |           0.053 |          0.662 |               0.821 |                0.774 |           0.001 |          0.002 |         0.031 |              0.010 |               0.027 |
| family_group  | ridge             | stability_ddG    |         840 |         0.457 |        0.355 |       0.241 |            0.513 |             0.507 |            0.437 |           0.343 |          0.234 |               0.573 |                0.569 |           0.056 |          0.041 |         0.140 |              0.068 |               0.062 |
| family_group  | ridge             | functional_score |         840 |         0.079 |        0.063 |       0.529 |            0.730 |             0.708 |            0.078 |           0.062 |          0.509 |               0.758 |                0.737 |           0.005 |          0.004 |         0.057 |              0.012 |               0.019 |
| family_group  | random_forest     | stability_ddG    |         840 |         0.442 |        0.346 |       0.291 |            0.548 |             0.522 |            0.422 |           0.334 |          0.284 |               0.608 |                0.581 |           0.055 |          0.041 |         0.153 |              0.055 |               0.050 |
| family_group  | random_forest     | functional_score |         840 |         0.073 |        0.058 |       0.592 |            0.776 |             0.721 |            0.072 |           0.057 |          0.587 |               0.803 |                0.756 |           0.007 |          0.006 |         0.077 |              0.035 |               0.034 |
| family_group  | gradient_boosting | stability_ddG    |         840 |         0.426 |        0.338 |       0.341 |            0.598 |             0.556 |            0.408 |           0.327 |          0.327 |               0.642 |                0.598 |           0.054 |          0.040 |         0.162 |              0.055 |               0.052 |
| family_group  | gradient_boosting | functional_score |         840 |         0.069 |        0.055 |       0.634 |            0.796 |             0.747 |            0.067 |           0.054 |          0.636 |               0.817 |                0.770 |           0.007 |          0.006 |         0.058 |              0.023 |               0.031 |

## Optimism gap

A positive gap is the amount by which a random split over-states performance relative to the grouped split, i.e. the credit the model gets for recognising proteins or families it has already seen.

| model             | target           | metric          |   random |   family_group |   optimism_gap |
|:------------------|:-----------------|:----------------|---------:|---------------:|---------------:|
| gradient_boosting | functional_score | pooled_spearman |    0.817 |          0.747 |          0.070 |
| gradient_boosting | stability_ddG    | pooled_spearman |    0.650 |          0.556 |          0.094 |
| random_forest     | functional_score | pooled_spearman |    0.792 |          0.721 |          0.072 |
| random_forest     | stability_ddG    | pooled_spearman |    0.627 |          0.522 |          0.105 |
| ridge             | functional_score | pooled_spearman |    0.769 |          0.708 |          0.061 |
| ridge             | stability_ddG    | pooled_spearman |    0.644 |          0.507 |          0.137 |

## Figures

- `pred_vs_actual.png`
- `residual_distribution.png`
- `strategy_comparison.png`


# Comparative evaluation

- Model families compared: linear, tree_ensemble_bagging, tree_ensemble_boosting, neural_network
- Splitting strategies compared: random, protein_group, family_group
- Reported strategy: `family_group`; ranking metric: `pooled_spearman`

`optimism_gap_<metric>` is the random split's score for that model and target minus this row's, so a positive gap is the credit the optimistic split awards for recognising proteins and families already seen in training.

## stability_ddG: family ranking under `family_group`

| model             | model_family           |   rank |   pooled_spearman |
|:------------------|:-----------------------|-------:|------------------:|
| gradient_boosting | tree_ensemble_boosting |      1 |             0.556 |
| multitask_nn      | neural_network         |      2 |             0.551 |
| random_forest     | tree_ensemble_bagging  |      3 |             0.522 |
| ridge             | linear                 |      4 |             0.507 |
| dual_branch_nn    | neural_network         |      5 |             0.390 |

## functional_score: family ranking under `family_group`

| model             | model_family           |   rank |   pooled_spearman |
|:------------------|:-----------------------|-------:|------------------:|
| gradient_boosting | tree_ensemble_boosting |      1 |             0.747 |
| random_forest     | tree_ensemble_bagging  |      2 |             0.721 |
| ridge             | linear                 |      3 |             0.708 |
| dual_branch_nn    | neural_network         |      4 |             0.282 |
| multitask_nn      | neural_network         |      5 |             0.218 |

## stability_ddG: all models, all strategies

| model             | model_family           | strategy      |   n_samples |   pooled_rmse |   fold_std_rmse |   optimism_gap_rmse |   pooled_mae |   fold_std_mae |   optimism_gap_mae |   pooled_r2 |   fold_std_r2 |   optimism_gap_r2 |   pooled_pearson |   fold_std_pearson |   optimism_gap_pearson |   pooled_spearman |   fold_std_spearman |   optimism_gap_spearman |
|:------------------|:-----------------------|:--------------|------------:|--------------:|----------------:|--------------------:|-------------:|---------------:|-------------------:|------------:|--------------:|------------------:|-----------------:|-------------------:|-----------------------:|------------------:|--------------------:|------------------------:|
| ridge             | linear                 | random        |         840 |         0.406 |           0.031 |             nan     |        0.315 |          0.017 |            nan     |       0.401 |         0.069 |           nan     |            0.634 |              0.055 |                nan     |             0.644 |               0.042 |                 nan     |
| ridge             | linear                 | protein_group |         840 |         0.411 |           0.015 |              -0.005 |        0.314 |          0.016 |              0.001 |       0.387 |         0.044 |             0.014 |            0.624 |              0.032 |                  0.010 |             0.637 |               0.030 |                   0.007 |
| ridge             | linear                 | family_group  |         840 |         0.457 |           0.056 |              -0.051 |        0.355 |          0.041 |             -0.041 |       0.241 |         0.140 |             0.160 |            0.513 |              0.068 |                  0.122 |             0.507 |               0.062 |                   0.137 |
| random_forest     | tree_ensemble_bagging  | random        |         840 |         0.409 |           0.034 |             nan     |        0.318 |          0.021 |            nan     |       0.392 |         0.063 |           nan     |            0.635 |              0.063 |                nan     |             0.627 |               0.065 |                 nan     |
| random_forest     | tree_ensemble_bagging  | protein_group |         840 |         0.411 |           0.023 |              -0.002 |        0.320 |          0.017 |             -0.002 |       0.386 |         0.042 |             0.005 |            0.634 |              0.042 |                  0.000 |             0.625 |               0.039 |                   0.002 |
| random_forest     | tree_ensemble_bagging  | family_group  |         840 |         0.442 |           0.055 |              -0.033 |        0.346 |          0.041 |             -0.029 |       0.291 |         0.153 |             0.101 |            0.548 |              0.055 |                  0.087 |             0.522 |               0.050 |                   0.105 |
| gradient_boosting | tree_ensemble_boosting | random        |         840 |         0.390 |           0.030 |             nan     |        0.311 |          0.023 |            nan     |       0.448 |         0.057 |           nan     |            0.669 |              0.045 |                nan     |             0.650 |               0.056 |                 nan     |
| gradient_boosting | tree_ensemble_boosting | protein_group |         840 |         0.387 |           0.007 |               0.002 |        0.308 |          0.006 |              0.003 |       0.454 |         0.061 |            -0.007 |            0.674 |              0.054 |                 -0.005 |             0.656 |               0.037 |                  -0.006 |
| gradient_boosting | tree_ensemble_boosting | family_group  |         840 |         0.426 |           0.054 |              -0.036 |        0.338 |          0.040 |             -0.027 |       0.341 |         0.162 |             0.107 |            0.598 |              0.055 |                  0.071 |             0.556 |               0.052 |                   0.094 |
| multitask_nn      | neural_network         | family_group  |         840 |         0.448 |           0.041 |             nan     |        0.346 |          0.030 |            nan     |       0.271 |         0.033 |           nan     |            0.531 |              0.046 |                nan     |             0.551 |               0.026 |                 nan     |
| dual_branch_nn    | neural_network         | family_group  |         840 |         0.495 |           0.070 |             nan     |        0.385 |          0.052 |            nan     |       0.110 |         0.218 |           nan     |            0.379 |              0.069 |                nan     |             0.390 |               0.050 |                 nan     |

## functional_score: all models, all strategies

| model             | model_family           | strategy      |   n_samples |   pooled_rmse |   fold_std_rmse |   optimism_gap_rmse |   pooled_mae |   fold_std_mae |   optimism_gap_mae |   pooled_r2 |   fold_std_r2 |   optimism_gap_r2 |   pooled_pearson |   fold_std_pearson |   optimism_gap_pearson |   pooled_spearman |   fold_std_spearman |   optimism_gap_spearman |
|:------------------|:-----------------------|:--------------|------------:|--------------:|----------------:|--------------------:|-------------:|---------------:|-------------------:|------------:|--------------:|------------------:|-----------------:|-------------------:|-----------------------:|------------------:|--------------------:|------------------------:|
| ridge             | linear                 | random        |         840 |         0.071 |           0.002 |             nan     |        0.057 |          0.002 |            nan     |       0.618 |         0.025 |           nan     |            0.787 |              0.017 |                nan     |             0.769 |               0.024 |                 nan     |
| ridge             | linear                 | protein_group |         840 |         0.072 |           0.003 |              -0.001 |        0.059 |          0.002 |             -0.001 |       0.602 |         0.064 |             0.016 |            0.776 |              0.037 |                  0.010 |             0.756 |               0.035 |                   0.013 |
| ridge             | linear                 | family_group  |         840 |         0.079 |           0.005 |              -0.008 |        0.063 |          0.004 |             -0.005 |       0.529 |         0.057 |             0.090 |            0.730 |              0.012 |                  0.057 |             0.708 |               0.019 |                   0.061 |
| random_forest     | tree_ensemble_bagging  | random        |         840 |         0.067 |           0.004 |             nan     |        0.053 |          0.003 |            nan     |       0.664 |         0.022 |           nan     |            0.826 |              0.014 |                nan     |             0.792 |               0.007 |                 nan     |
| random_forest     | tree_ensemble_bagging  | protein_group |         840 |         0.071 |           0.003 |              -0.004 |        0.057 |          0.003 |             -0.004 |       0.619 |         0.018 |             0.044 |            0.795 |              0.007 |                  0.031 |             0.746 |               0.025 |                   0.047 |
| random_forest     | tree_ensemble_bagging  | family_group  |         840 |         0.073 |           0.007 |              -0.007 |        0.058 |          0.006 |             -0.006 |       0.592 |         0.077 |             0.072 |            0.776 |              0.035 |                  0.050 |             0.721 |               0.034 |                   0.072 |
| gradient_boosting | tree_ensemble_boosting | random        |         840 |         0.060 |           0.002 |             nan     |        0.047 |          0.002 |            nan     |       0.724 |         0.020 |           nan     |            0.851 |              0.011 |                nan     |             0.817 |               0.011 |                 nan     |
| gradient_boosting | tree_ensemble_boosting | protein_group |         840 |         0.066 |           0.001 |              -0.006 |        0.053 |          0.002 |             -0.006 |       0.670 |         0.031 |             0.054 |            0.819 |              0.010 |                  0.032 |             0.771 |               0.027 |                   0.046 |
| gradient_boosting | tree_ensemble_boosting | family_group  |         840 |         0.069 |           0.007 |              -0.009 |        0.055 |          0.006 |             -0.008 |       0.634 |         0.058 |             0.090 |            0.796 |              0.023 |                  0.055 |             0.747 |               0.031 |                   0.070 |
| multitask_nn      | neural_network         | family_group  |         840 |         0.149 |           0.025 |             nan     |        0.126 |          0.023 |            nan     |      -0.677 |         0.516 |           nan     |            0.222 |              0.194 |                nan     |             0.218 |               0.208 |                 nan     |
| dual_branch_nn    | neural_network         | family_group  |         840 |         0.115 |           0.022 |             nan     |        0.091 |          0.022 |            nan     |      -0.002 |         0.337 |           nan     |            0.301 |              0.193 |                nan     |             0.282 |               0.211 |                 nan     |

## Interpretation

**Why the family-aware split scores lower.** Across every model the random split reports a higher `spearman` than `family_group`; the largest inflation is +0.137 for `ridge` on `stability_ddG`. A random split scatters mutations of the same protein across the train and validation sides, so a model can recognise the protein it is scoring and recall the average effect of mutating it. The family-grouped split withholds entire evolutionary families, so the model must extrapolate to a fold and a structural context it has never seen, and the family-level baseline effect it would otherwise lean on is unavailable. The grouped number is the honest estimate for the assignment's candidates, which are unlabelled proteins the model has not been trained on.

**Pearson vs Spearman.** Pearson measures how close the relationship is to a straight line and is therefore sensitive to a few badly-placed extreme values; Spearman measures only whether the ordering is right. The widest divergence here is `random_forest` on `functional_score`, at 0.776 Pearson against 0.721 Spearman. For this project Spearman is the one that matters: the deliverable is a ranked shortlist of candidates to take into the wet lab, so being right about which mutation is more promising counts for more than being numerically calibrated.

**Negative R².** `multitask_nn` on `functional_score`, `dual_branch_nn` on `functional_score` score below zero, meaning the predictions are worse than simply predicting the training mean. A negative R² alongside a positive rank correlation says the model has some signal about ordering but is badly mis-calibrated in magnitude; a negative R² with near-zero rank correlation says it has not learned the target at all.

