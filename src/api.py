"""HTTP service over the trained mutation-effect model.

Exposes the *inference* half of the pipeline — scoring mutations, ranking
candidates, reading the published metrics — as JSON endpoints that a client
such as Postman or curl can call.

Run it::

    python -m src.api --config config/config.yaml
    # or: uvicorn src.api:create_app --factory --reload

Then open ``http://127.0.0.1:8000/docs`` for the interactive schema, or import
``http://127.0.0.1:8000/openapi.json`` into Postman to get every endpoint as a
ready-made collection.

**What is deliberately not here: training.** There is no endpoint that runs
Phase 1, fits models or regenerates artifacts. A request that blocks for
minutes, mutates shared files and cannot be made idempotent is not an API; it
is a batch job with an HTTP trigger, and it would let two concurrent callers
corrupt each other's run. Training stays ``python main.py``. The service reads
what that produced and refuses to start if it is missing.

**The correctness property that matters.** A mutation posted to ``/predict``
goes through *the same* functions the batch pipeline uses — the same secondary
structure encoding, the same embedding join, the same delta engineering from
``src/data_processing.py``, and the same fold ensemble from
``src/predict.py``. Nothing is reimplemented for the request path, so the API
cannot drift from the published numbers. A test asserts that a candidate
scored over HTTP matches that candidate's row in
``results/mutation_rankings/candidate_predictions.csv`` exactly.

**The embedding is a lookup, not an input.** Callers send the mutation's own
fields; the 256-dimensional protein embedding is joined on ``protein_id`` from
the reference table. A protein absent from that table therefore cannot be
scored, and the service says so with a 422 rather than imputing a vector and
returning a confident number built on nothing.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from src.common import constants as const
from src.common.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, setup_logging
from src.common.io import read_table
from src.data_processing import encode_secondary_structure, engineer_features, merge_embeddings
from src.predict import (
    TrainedArtifacts,
    apply_viability_gate,
    build_top_k,
    combine_uncertainty,
    flag_distribution_shift,
    load_artifacts,
    predict_ensemble,
    score_candidates,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Request and response schemas
# =============================================================================
class MutationRequest(BaseModel):
    """One mutation to score.

    Mirrors a row of ``candidate_mutations_unlabeled.csv``. Every field the
    model consumes is required: a default would silently become a prediction
    about a value the caller never supplied. The engineered features (deltas,
    absolute deltas, relative position) are **not** accepted — the service
    computes them, so a caller cannot submit a delta inconsistent with the
    wild-type and mutant values it came from.
    """

    mutation_id: str = Field(..., description="Caller's identifier, echoed back in the response")
    protein_id: str = Field(..., description="Must exist in the protein embedding reference table")
    protein_family: str
    protein_length: int = Field(..., gt=0)
    wt_amino_acid: str = Field(..., min_length=1, max_length=1)
    position: int = Field(..., gt=0)
    mutant_amino_acid: str = Field(..., min_length=1, max_length=1)
    conservation_score: float = Field(..., ge=0.0, le=1.0)
    solvent_accessibility: float = Field(..., ge=0.0, le=1.0)
    secondary_structure: str
    distance_to_active_site: float = Field(..., ge=0.0)
    local_confidence_score: float
    wt_hydrophobicity: float
    mutant_hydrophobicity: float
    wt_charge: float
    mutant_charge: float
    wt_volume: float = Field(..., gt=0)
    mutant_volume: float = Field(..., gt=0)

    @field_validator("wt_amino_acid", "mutant_amino_acid")
    @classmethod
    def _uppercase_residue(cls, value: str) -> str:
        """Accept lower-case residues but normalise, since the lookups are upper-case."""
        return value.upper()

    @field_validator("position")
    @classmethod
    def _position_is_plausible(cls, value: int) -> int:
        """Reject a position that cannot exist; length is cross-checked below."""
        return value


class PredictionRequest(BaseModel):
    """A batch of mutations to score."""

    mutations: list[MutationRequest] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    """Model output for one mutation.

    ``*_uncertainty`` is the spread across the cross-validation fold models.
    It is a **lower bound** on predictive uncertainty: it captures where the
    folds disagree, and misses error they share, which is the dominant risk
    for a protein family absent from training. ``ood_flag`` is what tells you
    whether you are in that regime.
    """

    mutation_id: str
    protein_id: str
    protein_family: str
    predicted_stability_ddG: float = Field(..., description="kcal/mol; negative is stabilizing")
    stability_uncertainty: float = Field(..., description="Std across fold models")
    predicted_functional_score: float = Field(..., ge=0.0, le=1.0)
    function_uncertainty: float
    total_uncertainty: float
    ood_flag: str = Field(..., description="in_distribution | unseen_protein | unseen_family")
    viability: str = Field(..., description="pass | fails_stability | fails_function | fails_both")
    is_viable: bool


class RankedCandidate(PredictionResponse):
    """A scored mutation with its position in the prioritized shortlist."""

    priority_rank: int
    priority_score: float
    justification: str


class PrioritizationResponse(BaseModel):
    """The ranked shortlist plus the context needed to read it honestly."""

    n_submitted: int
    n_viable: int
    top_k: int
    candidates: list[RankedCandidate]


class HealthResponse(BaseModel):
    """Liveness and readiness in one payload."""

    status: str
    models_loaded: bool
    n_fold_models: dict[str, int]
    random_seed: int | None = None
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    """Provenance of the artifacts this process is serving."""

    selected_models: dict[str, str]
    primary_strategy: str
    random_seed: int
    n_fold_models: dict[str, int]
    n_tabular_features: int
    n_embedding_features: int
    versions: dict[str, str]


# =============================================================================
# Service state, loaded once
# =============================================================================
class ModelService:
    """Holds the trained artifacts and scores requests against them.

    Loaded once at start-up and shared by every request. Loading per request
    would re-read ten joblib pickles and re-parse the embedding table each
    time, turning a millisecond call into a slow one for no benefit: the
    artifacts are immutable for the lifetime of the process.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.artifacts: TrainedArtifacts = load_artifacts(cfg)
        self.embeddings = read_table(resolve_path(cfg, "raw_dir", "embeddings"))
        # Reference for distribution-shift flags: which proteins and families
        # the model was actually fitted on.
        self.train = read_table(resolve_path(cfg, "processed_dir", "processed_train"))
        self.known_proteins = set(self.embeddings[const.COL_PROTEIN_ID])
        logger.info(
            "Service ready: %d known proteins, %d training rows, seed %s",
            len(self.known_proteins),
            len(self.train),
            self.artifacts.metadata["random_seed"],
        )

    # -- Feature preparation ----------------------------------------------
    def prepare(self, mutations: list[MutationRequest]) -> pd.DataFrame:
        """Turn validated requests into the exact feature table the model expects.

        Runs the same three steps ``src/data_processing.py`` runs for the
        batch pipeline, in the same order, so a mutation scored here and the
        same mutation scored by ``python -m src.predict`` reach the model as
        identical rows.

        Raises:
            HTTPException: 422 if any protein has no embedding, naming them.
        """
        unknown = sorted({m.protein_id for m in mutations} - self.known_proteins)
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"No protein embedding for {unknown[:10]}"
                    f"{' and others' if len(unknown) > 10 else ''}. The 256-dimensional "
                    "embedding is looked up by protein_id and cannot be imputed; add the "
                    "protein to the embedding reference table and retrain."
                ),
            )
        frame = pd.DataFrame([m.model_dump() for m in mutations])
        over_length = frame[frame[const.COL_POSITION] > frame[const.COL_PROTEIN_LENGTH]]
        if not over_length.empty:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "position exceeds protein_length for "
                    f"{sorted(over_length[const.COL_MUTATION_ID])[:10]}"
                ),
            )

        name = "api_request"
        frame, _ = encode_secondary_structure(frame, self.cfg, name)
        frame = merge_embeddings(frame, self.embeddings, self.cfg, name)
        frame, _ = engineer_features(frame, self.cfg, name)

        missing = [c for c in self.artifacts.features.all if c not in frame.columns]
        if missing:
            # A server-side defect, not a client error: the request was valid
            # but this build cannot produce what the model was trained on.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prepared features are missing {len(missing)} columns, e.g. {missing[:5]}",
            )
        return frame

    # -- Scoring -----------------------------------------------------------
    def score(self, mutations: list[MutationRequest]) -> pd.DataFrame:
        """Predict, combine uncertainty, flag distribution shift and gate.

        Only the *absolute* quantities are computed here — the predictions,
        their combined spread, the OOD flag and the viability gate. Every
        batch-relative quantity (normalized objectives, priority score, rank)
        is added by :meth:`prioritize` instead, because it is meaningless for
        a request holding a single mutation.
        """
        frame = self.prepare(mutations)
        predicted = predict_ensemble(frame, self.artifacts, self.cfg)
        combined = combine_uncertainty(predicted)
        flagged = flag_distribution_shift(combined, self.train, self.cfg)
        return apply_viability_gate(flagged, self.cfg)

    def prioritize(self, mutations: list[MutationRequest]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score, then rank with the configured multi-objective priority score.

        Returns:
            ``(scored, shortlist)``. Normalisation is **within this request's
            batch**, exactly as the batch pipeline normalises within the
            candidate pool, so a one-mutation request gets a meaningless
            relative score — the absolute predictions and the gate are what
            carry meaning there.
        """
        scored = score_candidates(self.score(mutations), self.cfg)
        return scored, build_top_k(scored, self.cfg)


def to_prediction(row: pd.Series) -> PredictionResponse:
    """Map one scored row onto the response schema."""
    return PredictionResponse(
        mutation_id=str(row[const.COL_MUTATION_ID]),
        protein_id=str(row[const.COL_PROTEIN_ID]),
        protein_family=str(row[const.COL_PROTEIN_FAMILY]),
        predicted_stability_ddG=float(row[f"{const.PRED_PREFIX}{const.TARGET_STABILITY}"]),
        stability_uncertainty=float(row[f"{const.PRED_STD_PREFIX}{const.TARGET_STABILITY}"]),
        predicted_functional_score=float(row[f"{const.PRED_PREFIX}{const.TARGET_FUNCTION}"]),
        function_uncertainty=float(row[f"{const.PRED_STD_PREFIX}{const.TARGET_FUNCTION}"]),
        total_uncertainty=float(row[const.COL_TOTAL_UNCERTAINTY]),
        ood_flag=str(row[const.COL_OOD_FLAG]),
        viability=str(row[const.COL_VIABILITY]),
        is_viable=bool(row[const.COL_IS_VIABLE]),
    )


# =============================================================================
# Application
# =============================================================================
def get_service(request: Request) -> ModelService:
    """Dependency returning the loaded service, or 503 if start-up failed.

    A readiness failure is 503 rather than 500: the request was fine, the
    process is not ready to answer it.
    """
    service: ModelService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(getattr(request.app.state, "startup_error", "Model artifacts not loaded")),
        )
    return service


def create_app(cfg: dict[str, Any] | None = None) -> FastAPI:
    """Build the application.

    Args:
        cfg: Loaded configuration; read from the default path when omitted,
            which is what ``uvicorn src.api:create_app --factory`` does.
    """
    config = cfg if cfg is not None else load_config(DEFAULT_CONFIG_PATH)
    settings = config["api"]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load the model once at start-up; record the failure if it cannot be."""
        try:
            app.state.service = ModelService(config)
            app.state.startup_error = None
        except (FileNotFoundError, KeyError, ValueError) as error:
            # Do not crash the process: /health must stay reachable so a
            # deployment can report *why* it is unhealthy.
            app.state.service = None
            app.state.startup_error = error
            logger.error("Model artifacts could not be loaded: %s", error)
        yield

    app = FastAPI(
        title=settings["title"],
        version=str(config["project"].get("version", "1.0.0")),
        description=__doc__.split("\n\n")[0],
        lifespan=lifespan,
    )
    if settings["cors_allow_origins"]:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings["cors_allow_origins"]),
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    max_batch = int(settings["max_batch_size"])

    def check_batch(mutations: list[MutationRequest]) -> None:
        """Reject a batch too large to serve in one request."""
        if len(mutations) > max_batch:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"{len(mutations)} mutations exceeds api.max_batch_size={max_batch}",
            )

    @app.get(const.API_HEALTH_PATH, response_model=HealthResponse, tags=["service"])
    def health(request: Request) -> HealthResponse:
        """Liveness and readiness. Always answers, even when the model is absent."""
        service: ModelService | None = getattr(request.app.state, "service", None)
        if service is None:
            return HealthResponse(
                status=const.API_STATUS_DEGRADED,
                models_loaded=False,
                n_fold_models={},
                detail=str(getattr(request.app.state, "startup_error", "not loaded")),
            )
        return HealthResponse(
            status=const.API_STATUS_OK,
            models_loaded=True,
            n_fold_models={t: len(p) for t, p in service.artifacts.fold_pipelines.items()},
            random_seed=int(service.artifacts.metadata["random_seed"]),
        )

    @app.get(const.API_MODEL_PATH, response_model=ModelInfoResponse, tags=["service"])
    def model_info(service: ModelService = Depends(get_service)) -> ModelInfoResponse:
        """What is being served: models, CV strategy, seed and library versions."""
        metadata = service.artifacts.metadata
        return ModelInfoResponse(
            selected_models={
                target: choice[const.COL_MODEL]
                for target, choice in service.artifacts.selection.items()
            },
            primary_strategy=str(metadata["primary_strategy"]),
            random_seed=int(metadata["random_seed"]),
            n_fold_models={t: len(p) for t, p in service.artifacts.fold_pipelines.items()},
            n_tabular_features=len(service.artifacts.features.tabular),
            n_embedding_features=len(service.artifacts.features.embedding),
            versions={k: str(v) for k, v in metadata.get("versions", {}).items()},
        )

    @app.post(
        const.API_PREDICT_PATH, response_model=list[PredictionResponse], tags=["inference"]
    )
    def predict(
        payload: PredictionRequest, service: ModelService = Depends(get_service)
    ) -> list[PredictionResponse]:
        """Score mutations: both predictions, fold-ensemble spread, OOD and viability.

        Order is preserved, so the Nth response corresponds to the Nth
        submitted mutation.
        """
        check_batch(payload.mutations)
        scored = service.score(payload.mutations)
        return [to_prediction(row) for _, row in scored.iterrows()]

    @app.post(
        const.API_PRIORITIZE_PATH, response_model=PrioritizationResponse, tags=["inference"]
    )
    def prioritize(
        payload: PredictionRequest, service: ModelService = Depends(get_service)
    ) -> PrioritizationResponse:
        """Score and rank a batch, returning the configured Top-K with justifications.

        The priority score is normalised **within the submitted batch**, so it
        ranks these mutations against each other and says nothing about their
        absolute merit. ``is_viable`` is the absolute judgement.
        """
        check_batch(payload.mutations)
        scored, shortlist = service.prioritize(payload.mutations)
        # drop=False: ``to_prediction`` reads the id back off the row, so the
        # column has to survive becoming the index.
        lookup = scored.set_index(const.COL_MUTATION_ID, drop=False)
        candidates = [
            RankedCandidate(
                **to_prediction(lookup.loc[row[const.COL_MUTATION_ID]]).model_dump(),
                priority_rank=int(row[const.COL_PRIORITY_RANK]),
                priority_score=float(row[const.COL_PRIORITY_SCORE]),
                justification=str(row[const.COL_JUSTIFICATION]),
            )
            for _, row in shortlist.iterrows()
        ]
        return PrioritizationResponse(
            n_submitted=len(payload.mutations),
            n_viable=int(scored[const.COL_IS_VIABLE].sum()),
            top_k=len(candidates),
            candidates=candidates,
        )

    @app.get(f"{const.API_METRICS_PATH}", tags=["results"])
    def list_metrics() -> dict[str, list[str]]:
        """Names of the metric tables this service will serve."""
        return {"available": list(const.API_SERVABLE_METRICS)}

    @app.get(f"{const.API_METRICS_PATH}/{{name}}", tags=["results"])
    def get_metrics(name: str) -> list[dict[str, Any]]:
        """Return one published metrics table as JSON records.

        ``name`` is checked against an allowlist, so the path parameter cannot
        be used to read a file outside the metrics directory.
        """
        if name not in const.API_SERVABLE_METRICS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown table '{name}'; available: {list(const.API_SERVABLE_METRICS)}",
            )
        path = resolve_path(config, "metrics_dir", name)
        if not path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{name} has not been generated yet; run the pipeline stage that writes it",
            )
        table = read_table(path)
        records: list[dict[str, Any]] = table.to_dict(orient="records")
        return records

    if settings["serve_figures"]:

        @app.get(f"{const.API_FIGURES_PATH}/{{key}}", tags=["results"])
        def get_figure(key: str) -> FileResponse:
            """Serve one generated PNG, addressed by its config ``files`` key."""
            known = (
                *const.FIGURE_FILE_KEYS,
                *const.TORCH_FIGURE_FILE_KEYS,
                *const.TORCH_V2_FIGURE_FILE_KEYS,
                *const.ABLATION_FIGURE_FILE_KEYS,
                *const.ATTR_FIGURE_FILE_KEYS,
            )
            if key not in known:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Unknown figure '{key}'; available: {list(known)}",
                )
            path = resolve_path(config, "figures_dir", key)
            if not path.is_file():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"{key} has not been generated yet",
                )
            return FileResponse(path, media_type="image/png")

    return app


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--host", default=None, help="Override api.host")
    parser.add_argument("--port", type=int, default=None, help="Override api.port")
    parser.add_argument("--reload", action="store_true", help="Reload on source changes")
    return parser.parse_args()


def main() -> None:
    """Run the service with uvicorn, using the configured host and port."""
    import uvicorn

    args = parse_args()
    config = load_config(args.config)
    setup_logging(config)
    settings = config["api"]
    uvicorn.run(
        create_app(config),
        host=args.host or settings["host"],
        port=args.port or int(settings["port"]),
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
