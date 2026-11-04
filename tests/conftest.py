"""Shared pytest fixtures: a validated config and small synthetic datasets.

The synthetic tables mirror the schema of the real raw CSVs (column names come
from ``src.common.constants``) but are tiny and generated from the project seed,
so tests are fast, deterministic and never touch ``data/`` or ``outputs/``.

Layout of the synthetic data:
    * 4 proteins in 2 families (PF_A: P1, P2; PF_B: P3, P4) with 256-d embeddings.
    * Labeled mutations on P1-P3 only.
    * Candidate mutations on P1-P4, so P4 / PF_B-only proteins test the
      unseen-protein reporting path.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.common import constants as const
from src.common.config import load_config

# Synthetic test-data layout (test data, not domain constants).
PROTEINS: dict[str, tuple[str, int]] = {  # protein_id -> (family, length)
    "P1": ("PF_A", 300),
    "P2": ("PF_A", 420),
    "P3": ("PF_B", 250),
    "P4": ("PF_B", 510),
}
LABELED_PROTEINS: tuple[str, ...] = ("P1", "P2", "P3")
CANDIDATE_PROTEINS: tuple[str, ...] = ("P1", "P2", "P3", "P4")
N_LABELED: int = 24
N_CANDIDATES: int = 12
AMINO_ACIDS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
CHARGES: tuple[float, ...] = (-1.0, 0.0, 1.0)


@pytest.fixture(scope="session")
def base_config() -> dict[str, Any]:
    """Load the real project config once; tests must deep-copy before mutating."""
    return load_config()


@pytest.fixture()
def cfg(base_config: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Return a per-test config whose every directory points inside ``tmp_path``.

    ``resolve_path`` joins ``PROJECT_ROOT / <path>``; an absolute path on the
    right-hand side replaces the root, so all I/O is redirected to ``tmp_path``.
    """
    config = copy.deepcopy(base_config)
    config["paths"] = {key: str(tmp_path / rel) for key, rel in base_config["paths"].items()}
    return config


@pytest.fixture()
def seed(base_config: dict[str, Any]) -> int:
    """Project random seed (20261102), read from config rather than repeated here."""
    return int(base_config["project"]["random_seed"])


@pytest.fixture()
def rng(seed: int) -> np.random.Generator:
    """Fresh seeded generator per test so tests are order-independent."""
    return np.random.default_rng(seed)


def make_embeddings(rng: np.random.Generator) -> pd.DataFrame:
    """Build a ``protein_embeddings.csv``-shaped table (one row per protein)."""
    ids = list(PROTEINS)
    df = pd.DataFrame(
        {
            const.COL_PROTEIN_ID: ids,
            const.COL_PROTEIN_FAMILY: [PROTEINS[p][0] for p in ids],
            const.COL_PROTEIN_LENGTH: [PROTEINS[p][1] for p in ids],
        }
    )
    emb = pd.DataFrame(
        rng.normal(size=(len(ids), const.EMBEDDING_DIM)), columns=list(const.EMBEDDING_COLUMNS)
    )
    return pd.concat([df, emb], axis=1)


def make_mutations(
    rng: np.random.Generator,
    n_rows: int,
    proteins: tuple[str, ...],
    id_prefix: str,
    with_targets: bool,
) -> pd.DataFrame:
    """Build a mutation table matching the labeled / candidate CSV schema.

    Every protein in ``proteins`` gets at least one row and every secondary
    structure code appears at least once, so encoders see all categories.
    """
    protein_ids = [proteins[i % len(proteins)] for i in range(n_rows)]
    ss_codes = [const.SS_CATEGORIES[i % len(const.SS_CATEGORIES)] for i in range(n_rows)]
    lengths = np.array([PROTEINS[p][1] for p in protein_ids])
    df = pd.DataFrame(
        {
            const.COL_MUTATION_ID: [f"{id_prefix}{i:04d}" for i in range(1, n_rows + 1)],
            const.COL_PROTEIN_ID: protein_ids,
            const.COL_PROTEIN_FAMILY: [PROTEINS[p][0] for p in protein_ids],
            const.COL_PROTEIN_LENGTH: lengths,
            const.COL_WT_AA: rng.choice(AMINO_ACIDS, size=n_rows),
            const.COL_POSITION: rng.integers(1, lengths + 1),
            const.COL_MUTANT_AA: rng.choice(AMINO_ACIDS, size=n_rows),
            const.COL_CONSERVATION: rng.uniform(0.0, 1.0, n_rows),
            const.COL_SOLVENT_ACCESSIBILITY: rng.uniform(0.0, 1.0, n_rows),
            const.COL_SECONDARY_STRUCTURE: ss_codes,
            const.COL_DIST_ACTIVE_SITE: rng.uniform(0.0, 40.0, n_rows),
            const.COL_LOCAL_CONFIDENCE: rng.uniform(50.0, 99.0, n_rows),
            const.COL_WT_HYDROPHOBICITY: rng.uniform(-4.5, 4.5, n_rows),
            const.COL_MUT_HYDROPHOBICITY: rng.uniform(-4.5, 4.5, n_rows),
            const.COL_WT_CHARGE: rng.choice(CHARGES, size=n_rows),
            const.COL_MUT_CHARGE: rng.choice(CHARGES, size=n_rows),
            const.COL_WT_VOLUME: rng.uniform(60.0, 230.0, n_rows),
            const.COL_MUT_VOLUME: rng.uniform(60.0, 230.0, n_rows),
        }
    )
    if with_targets:
        df[const.TARGET_STABILITY] = rng.normal(0.5, 1.0, n_rows)
        df[const.TARGET_FUNCTION] = rng.uniform(0.0, 1.0, n_rows)
    return df


@pytest.fixture()
def embeddings(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic protein embedding table."""
    return make_embeddings(rng)


@pytest.fixture()
def labeled(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic labeled mutations (with both targets)."""
    return make_mutations(rng, N_LABELED, LABELED_PROTEINS, "MUT", with_targets=True)


@pytest.fixture()
def candidates(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic unlabeled candidate mutations, including an unseen protein."""
    return make_mutations(rng, N_CANDIDATES, CANDIDATE_PROTEINS, "CAND", with_targets=False)


@pytest.fixture()
def frames(
    labeled: pd.DataFrame, candidates: pd.DataFrame, embeddings: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """The three raw tables keyed exactly as ``load_raw_data`` returns them."""
    return {
        const.TABLE_LABELED: labeled,
        const.TABLE_CANDIDATES: candidates,
        const.TABLE_EMBEDDINGS: embeddings,
    }


@pytest.fixture()
def raw_files(cfg: dict[str, Any], frames: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """Write the synthetic tables as CSVs into the temporary raw directory."""
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True)
    paths = {}
    for key, df in frames.items():
        paths[key] = raw_dir / cfg["files"][key]
        df.to_csv(paths[key], index=False)
    return paths
