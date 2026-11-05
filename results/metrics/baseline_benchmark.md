# Baseline model benchmark

- Models: ridge, random_forest, gradient_boosting
- Targets: stability_ddG, functional_score, modelled independently
- Primary strategy: `family_group`; selection metric: `pooled_spearman`

All metrics are pooled over every out-of-fold prediction. Each model is re-fitted from scratch inside every training fold, with imputation, scaling and embedding PCA as steps **inside** the pipeline, so no fold sees its validation rows at fit time.

`optimism_gap` is the random split's `pooled_spearman` minus this row's: the credit a model gets for recognising proteins and families it has already seen.

## stability_ddG

| model             | strategy      |   n_samples |   pooled_rmse |   pooled_mae |   pooled_r2 |   pooled_pearson |   pooled_spearman |   fold_std_spearman |   optimism_gap |
|:------------------|:--------------|------------:|--------------:|-------------:|------------:|-----------------:|------------------:|--------------------:|---------------:|
| ridge             | random        |         840 |         0.406 |        0.315 |       0.401 |            0.634 |             0.644 |               0.042 |        nan     |
| ridge             | protein_group |         840 |         0.411 |        0.314 |       0.387 |            0.624 |             0.637 |               0.030 |          0.007 |
| ridge             | family_group  |         840 |         0.457 |        0.355 |       0.241 |            0.513 |             0.507 |               0.062 |          0.137 |
| random_forest     | random        |         840 |         0.409 |        0.318 |       0.392 |            0.635 |             0.627 |               0.065 |        nan     |
| random_forest     | protein_group |         840 |         0.411 |        0.320 |       0.386 |            0.634 |             0.625 |               0.039 |          0.002 |
| random_forest     | family_group  |         840 |         0.442 |        0.346 |       0.291 |            0.548 |             0.522 |               0.050 |          0.105 |
| gradient_boosting | random        |         840 |         0.390 |        0.311 |       0.448 |            0.669 |             0.650 |               0.056 |        nan     |
| gradient_boosting | protein_group |         840 |         0.387 |        0.308 |       0.454 |            0.674 |             0.656 |               0.037 |         -0.006 |
| gradient_boosting | family_group  |         840 |         0.426 |        0.338 |       0.341 |            0.598 |             0.556 |               0.052 |          0.094 |

## functional_score

| model             | strategy      |   n_samples |   pooled_rmse |   pooled_mae |   pooled_r2 |   pooled_pearson |   pooled_spearman |   fold_std_spearman |   optimism_gap |
|:------------------|:--------------|------------:|--------------:|-------------:|------------:|-----------------:|------------------:|--------------------:|---------------:|
| ridge             | random        |         840 |         0.071 |        0.057 |       0.618 |            0.787 |             0.769 |               0.024 |        nan     |
| ridge             | protein_group |         840 |         0.072 |        0.059 |       0.602 |            0.776 |             0.756 |               0.035 |          0.013 |
| ridge             | family_group  |         840 |         0.079 |        0.063 |       0.529 |            0.730 |             0.708 |               0.019 |          0.061 |
| random_forest     | random        |         840 |         0.067 |        0.053 |       0.664 |            0.826 |             0.792 |               0.007 |        nan     |
| random_forest     | protein_group |         840 |         0.071 |        0.057 |       0.619 |            0.795 |             0.746 |               0.025 |          0.047 |
| random_forest     | family_group  |         840 |         0.073 |        0.058 |       0.592 |            0.776 |             0.721 |               0.034 |          0.072 |
| gradient_boosting | random        |         840 |         0.060 |        0.047 |       0.724 |            0.851 |             0.817 |               0.011 |        nan     |
| gradient_boosting | protein_group |         840 |         0.066 |        0.053 |       0.670 |            0.819 |             0.771 |               0.027 |          0.046 |
| gradient_boosting | family_group  |         840 |         0.069 |        0.055 |       0.634 |            0.796 |             0.747 |               0.031 |          0.070 |

