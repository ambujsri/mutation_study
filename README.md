# Medvolt.AI — Protein Mutation-Effect Prediction & Candidate Prioritization

## Problem statement

Given 840 experimentally characterised single-point mutations across 28 proteins, predict
two effects of a mutation and use them to prioritise 180 unlabeled candidates for
wet-lab validation:

| Target | Meaning | Direction |
|---|---|---|
| `stability_ddG` | Change in folding free energy (kcal/mol) | **Negative is better** (stabilizing) |
| `functional_score` | Retained activity, bounded | **Higher is better** |

The deliverable is a ranked Top 10 with a calibrated sense of how much each ranking can
be trusted. The engineering problem underneath it is **leakage**: mutations are nested
inside proteins, which are nested inside families, and 32% of the candidate pool comes
from two families (`PF08`, `PF09`) that appear nowhere in training. A model validated on
a random split is scored largely on proteins it has already seen, and would therefore
over-promise on exactly the candidates that matter most.

## Dataset

| | Rows | Proteins | Families |
|---|---|---|---|
| `labeled_mutations.csv` | 840 | 28 | 7 (PF01–PF07) |
| `candidate_mutations_unlabeled.csv` | 180 | 35 (8 unseen) | 9 (PF08, PF09 unseen) |
| `protein_embeddings.csv` | 36 | 36 | 9 |

Per-mutation features cover the wild-type and mutant residue (hydrophobicity, charge,
volume), the structural microenvironment (conservation, solvent accessibility, distance
to the active site, secondary structure, local confidence) and sequence position. Each
protein also carries a 256-dimensional pretrained embedding, which is **constant within a
protein** — a fact with consequences, quantified in the ablation below.

The raw data contains no missing, infinite or out-of-range values; this is verified on
every run rather than assumed (`src/data_processing.py`, `inspect_anomalies`).

## Setup

Requires **Python 3.11 or newer**. Python 3.10 will not work: the newest scipy that
supports it (1.15.x) ships macOS arm64 wheels whose compiled extensions fail to load
on macOS 26+, and the fix requires Python >= 3.11. See the note in
[requirements.txt](requirements.txt).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or requirements-dev.txt for lint/type/test tooling
```

## How to run

The whole pipeline, end to end:

```bash
python main.py --config config/config.yaml
```

Selected stages, always executed in dependency order:

```bash
python main.py --list                 # show the stage names
python main.py --stages train evaluate
python main.py --from evaluate        # this stage and everything after it
```

Every stage is also runnable on its own, which is useful when iterating on one of them:

```bash
python -m src.data_processing --config config/config.yaml   # merge, clean, engineer features
python -m src.validation      --config config/config.yaml   # CV fold report and leakage audit
python -m src.train           --config config/config.yaml   # train, select, save models
python -m src.evaluate        --config config/config.yaml   # metrics report and figures
python -m src.predict         --config config/config.yaml   # score candidates, Top 10
python -m src.ablation        --config config/config.yaml   # feature-subset ablation
python -m src.interpret       --config config/config.yaml   # per-candidate write-up
```

Training fits 90 pipelines. Runtime depends heavily on the BLAS behind numpy: roughly
1 minute with Anaconda's MKL build, but around 15 minutes with the default
Homebrew/PyPI build.

Checks:

```bash
python -m pytest -q        # 487 tests
python -m ruff check .
python -m mypy src main.py
```

## Model architecture

Every model is a single scikit-learn `Pipeline`, built in [src/model.py](src/model.py):

```text
ColumnTransformer
  ├─ tabular   : impute → scale                 (22 engineered + raw features)
  └─ embedding : impute → scale → PCA(16)       (256-d protein embedding)
→ regressor   (ridge | random_forest | gradient_boosting)
```

Keeping imputation, scaling and PCA **inside** the pipeline is what makes leakage control
structural rather than procedural: `fit` on a training fold is the only place those
statistics are learned, and the validation fold is only ever passed to `predict`. The
tests assert this directly — fitted statistics must equal the training fold's and
*differ* from the whole table's, and corrupting validation rows must leave every fitted
object bit-identical.

Both targets are modelled independently; the model for each is chosen on its own merits.

## Validation approach

Three strategies are run on every model and target
([src/validation.py](src/validation.py)):

| Strategy | Group | Question it answers |
|---|---|---|
| `random` | none | Optimistic reference only |
| `protein_group` | `protein_id` | Generalisation to unseen proteins |
| `family_group` | `protein_family` | **Primary** — generalisation to unseen families |

`family_group` is primary because it is the only strategy whose held-out folds resemble
the candidate pool. Metrics are reported both per fold (mean ± std) and **pooled** over
all out-of-fold predictions; pooled is the headline, because family folds are small and
uneven (1–2 families each) so per-fold means are noisy.

The gap between the two is reported explicitly as the **optimism gap**:

| Model | Target | random | family_group | Gap |
|---|---|---|---|---|
| gradient_boosting | `stability_ddG` | 0.650 | 0.556 | **+0.094** |
| gradient_boosting | `functional_score` | 0.817 | 0.747 | **+0.070** |
| ridge | `stability_ddG` | 0.644 | 0.507 | **+0.137** |

(pooled Spearman; full table in `results/metrics/evaluation_summary.md`.)

## Main results

Selected model for both targets: **gradient boosting**, by pooled Spearman under
family-grouped CV.

| Target | Pooled Spearman (family_group) |
|---|---|
| `stability_ddG` | 0.556 |
| `functional_score` | 0.747 |

**Ablation — the headline scientific finding.** The 256-d embeddings contribute almost
nothing, and on their own are worse than useless:

| Subset | Features | `stability_ddG` | `functional_score` |
|---|---|---|---|
| `D_full` (baseline) | 22 + 256 | 0.556 | 0.747 |
| `E_all_tabular_no_embeddings` | 22 + 0 | 0.546 (−0.010) | 0.723 (−0.025) |
| `G_full_minus_structural` | 15 + 256 | 0.413 (−0.143) | 0.370 (−0.377) |
| `B_embeddings_only` | 0 + 256 | **−0.205** | **−0.110** |

Because the embedding is a per-protein constant, it cannot distinguish one mutation from
another *within* a protein — which is exactly the ranking being asked for. Structural
features carry the signal. A 22-feature model with no embeddings and no PCA matches the
278-feature model to within 0.01–0.03.

**Uncertainty — a negative result worth stating.** Ensemble spread is *lower* for
unseen-family candidates (0.0969 vs 0.1031 for stability), so the uncertainty penalty
does **not** protect against domain shift. Only the explicit `ood_flag` does.

**Honest caveat on the Top 10.** Only 5 of 180 candidates are predicted stabilizing, so
the stability objective mostly selects the *least destabilizing* mutations; and predicted
function spans only 0.798–0.905 across the Top 10, narrower than the target's own RMSE.
Single-rank ordering within the Top 10 should not be over-read.

Full analysis: [report/final_report.pdf](report/final_report.pdf) (6 pages), with the
extended version in [report/final_report_full.md](report/final_report_full.md).

## Repository structure

```text
config/config.yaml              Run setup and decisions: paths, seed, CV, models, ranking
data/raw/                       Original CSVs (read-only inputs)
data/processed/                 Engineered train / candidate tables + feature manifest
notebooks/
  exploratory_analysis.ipynb    EDA: completeness, duplicates, distributions, outliers,
                                mutation types, correlations, data-quality findings
src/common/                     constants.py (dataset facts, naming), config.py, io.py
src/data_processing.py          Ingestion, merge with embeddings, anomaly checks, features
src/model.py                    Model architectures: feature blocks, pipelines, registry
src/validation.py               Random vs. protein-group vs. family-group CV
src/train.py                    Fold-wise training, model selection, artifacts
src/evaluate.py                 Metrics, optimism gap, figures, evaluation report
src/predict.py                  Ensemble inference, uncertainty, OOD flags, Top 10
src/ablation.py                 Feature-subset ablation study
src/interpret.py                Per-candidate scientific interpretation
results/metrics/                CV metrics, OOF predictions, ablation, evaluation summary
results/figures/                Evaluation plots
results/mutation_rankings/      Scored candidates, Top 10, candidate interpretation
results/models/                 Fitted pipelines + training_metadata.json (provenance)
report/final_report.md          Source for the 6-page technical report
report/final_report.pdf         The report deliverable (built by report/build_pdf.py)
report/final_report_full.md     Extended version: full derivations, all candidate write-ups
tests/                          487 tests, mirroring src/
main.py                         End-to-end orchestration
docs/CODING_GUIDELINES.md       Standards every change must follow
```

Contributors: read [docs/CODING_GUIDELINES.md](docs/CODING_GUIDELINES.md) first. Section 4
defines what goes in `config.yaml` versus `src/common/constants.py`.

## Reproducibility

Random seed: `20261102`, set once in `config/config.yaml` and threaded through every
splitter, estimator and PCA.

Results are reproducible for a fixed environment, but **not across library versions**.
Re-running any stage with the pinned versions reproduces its outputs exactly; changing
the scikit-learn pin shifts predictions slightly and reorders the Top 10 (overlap stays
9/10). Model files in `results/models/` are joblib pickles and cannot be loaded by a
different scikit-learn version, so changing that pin requires re-running training before
prediction. The exact versions, seed and feature lists used are recorded in
`results/models/training_metadata.json`.
