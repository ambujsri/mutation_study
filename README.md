# Medvolt.AI — Protein Mutation-Effect Prediction & Candidate Prioritization

Predict `stability_ddG` and `functional_score` for single-point mutations, then rank
180 unlabeled candidates and report a Top 10.

## Layout

```text
config/config.yaml   Run setup and decisions: paths, seed, encoding, CV, models, ranking
data/raw/            Original CSVs (read-only inputs)
data/processed/      Engineered train / candidate tables + feature manifest
src/common/          constants.py (dataset facts, naming) and config.py (load/validate config)
src/preprocess.py    Loading, merge with embeddings, anomaly checks, delta features
src/validation.py    Random vs. protein-group vs. family-group CV
src/train.py         Multi-target training and CV logging
src/predict.py       Inference, multi-objective scoring, Top 10
outputs/             models/, figures/, reports/
main.py              End-to-end orchestration
docs/                CODING_GUIDELINES.md — standards every change must follow
```

Contributors: read [docs/CODING_GUIDELINES.md](docs/CODING_GUIDELINES.md) first. Section 4
defines what goes in `config.yaml` versus `src/common/constants.py`.

## Setup

Requires **Python 3.11 or newer**. Python 3.10 will not work: the newest scipy that
supports it (1.15.x) ships macOS arm64 wheels whose compiled extensions fail to load
on macOS 26+, and the fix requires Python >= 3.11. See the note in
[requirements.txt](requirements.txt).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or requirements-dev.txt for lint/type/test tooling
```

## Run

Each phase can run on its own, in this order:

```bash
python -m src.preprocess --config config/config.yaml   # Phase 1: merge, clean, engineer
python -m src.validation --config config/config.yaml   # Phase 2: CV fold report (optional)
python -m src.train      --config config/config.yaml   # Phase 3: train, select, save models
python -m src.predict    --config config/config.yaml   # Phase 4: score candidates, Top 10
```

Phase 3 fits 90 pipelines. Runtime depends heavily on the BLAS behind numpy:
roughly 1 minute with Anaconda's MKL build, but around 15 minutes with the
default Homebrew/PyPI build.

## Reproducibility

Results are reproducible for a fixed environment, but **not across library versions**.
Re-running any phase with the pinned versions reproduces its outputs exactly; changing
the scikit-learn pin shifts predictions slightly and reorders the Top 10. Model files in
`outputs/models/` are joblib pickles and cannot be loaded by a different scikit-learn
version, so changing that pin requires re-running Phase 3 before Phase 4. The exact
versions, seed and feature lists used are recorded in
`outputs/models/training_metadata.json`.

## Dataset notes

| | Rows | Proteins | Families |
|---|---|---|---|
| Labeled | 840 | 28 | 7 (PF01–PF07) |
| Candidates | 180 | 35 (8 unseen) | 9 (PF08, PF09 unseen) |
| Embeddings | 36 | 36 | 9 |

Candidates include two protein families absent from training, so family-aware
grouped CV is the primary validation strategy; random splits are reported only as an
optimistic reference.

Random seed: `20261102` (set in `config/config.yaml`).
