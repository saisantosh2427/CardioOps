"""Stage 7 - FastAPI service that serves the registry champion.

Endpoints: ``GET /health``, ``POST /predict``, ``POST /predict/batch``,
``GET /model-info``, ``POST /feedback`` (delayed ground truth),
``POST /model/reload`` and ``GET /metrics`` (Prometheus).

Every prediction is written to the prediction store so that monitoring and
retraining operate on real served traffic. A background task polls the
registry and hot-swaps the model when the ``champion`` alias moves.

Run: ``uvicorn api.main:app --host 0.0.0.0 --port 8000``
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api import metrics
from api.model_service import LoadedModel, ModelService, ModelUnavailableError
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PatientFeatures,
    PredictionResponse,
)
from src import __version__
from src.columns import FEATURE_COLUMNS
from src.config import Settings, get_settings
from src.logging_utils import configure_logging, get_logger
from src.store import PredictionRecord, PredictionStore

logger = get_logger("api")


def _refresh_model(service: ModelService) -> bool:
    """Refresh the served model and keep Prometheus gauges in sync."""
    changed = service.refresh()
    current = service.current
    if current is None:
        metrics.MODEL_LOADED.set(0)
    elif changed:
        metrics.MODEL_RELOADS.inc()
        metrics.set_model_info(current.name, current.version, current.model_type)
    return changed


async def _refresh_loop(service: ModelService, interval: int) -> None:
    while True:
        await asyncio.sleep(interval)
        await asyncio.to_thread(_refresh_model, service)


def _get_store(app: FastAPI) -> PredictionStore | None:
    """Return the prediction store, connecting lazily (and retrying) if needed."""
    if app.state.store is None:
        settings: Settings = app.state.settings
        try:
            app.state.store = PredictionStore(settings.database_url)
        except Exception as exc:
            logger.error("prediction_store_unavailable", error=str(exc))
            return None
    return app.state.store


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory (tests pass their own settings)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.project.log_level, settings.project.log_format)
        app.state.started_at = time.monotonic()
        _get_store(app)
        await asyncio.to_thread(_refresh_model, app.state.model_service)
        refresher = None
        if settings.api.refresh_interval_seconds > 0:
            refresher = asyncio.create_task(
                _refresh_loop(app.state.model_service, settings.api.refresh_interval_seconds)
            )
        logger.info("api_started", model_uri=settings.champion_model_uri)
        yield
        if refresher is not None:
            refresher.cancel()
            with suppress(asyncio.CancelledError):
                await refresher
        if app.state.store is not None:
            app.state.store.dispose()

    app = FastAPI(
        title="Heart Disease Risk API",
        version=__version__,
        description=(
            "Predicts the probability of heart disease from clinical measurements "
            "(UCI Heart Disease). Serves the MLflow registry `champion` model. "
            "**Not a medical device** - for MLOps demonstration only."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_service = ModelService(settings)
    app.state.store = None
    app.state.started_at = time.monotonic()

    @app.middleware("http")
    async def observe_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - started
            route = getattr(request.scope.get("route"), "path", "unmatched")
            if route != "/metrics":
                metrics.HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
                metrics.HTTP_LATENCY.labels(request.method, route).observe(elapsed)
            if route not in ("/metrics", "/health"):
                logger.info(
                    "http_request",
                    method=request.method,
                    route=route,
                    status=status,
                    duration_ms=round(elapsed * 1000, 2),
                )

    def run_predictions(
        patients: list[PatientFeatures],
    ) -> tuple[list[PredictionResponse], LoadedModel]:
        started = time.perf_counter()
        payloads = [p.model_dump() for p in patients]
        frame = pd.DataFrame(payloads, columns=FEATURE_COLUMNS)
        try:
            proba, loaded = app.state.model_service.predict_proba(frame)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
        threshold = settings.training.decision_threshold
        latency_ms = (time.perf_counter() - started) * 1000 / len(patients)

        responses, records = [], []
        for payload, p in zip(payloads, proba, strict=True):
            prediction = int(p >= threshold)
            request_id = str(uuid.uuid4())
            responses.append(
                PredictionResponse(
                    request_id=request_id,
                    prediction=prediction,
                    probability=round(float(p), 6),
                    risk_label="high" if prediction else "low",
                    model_name=loaded.name,
                    model_version=loaded.version,
                )
            )
            records.append(
                PredictionRecord(
                    request_id=request_id,
                    model_name=loaded.name,
                    model_version=loaded.version,
                    features=payload,
                    prediction=prediction,
                    probability=float(p),
                    latency_ms=latency_ms,
                )
            )
            metrics.PREDICTIONS.labels(loaded.version, str(prediction)).inc()
            metrics.PREDICTED_PROBABILITY.observe(float(p))

        # Logged synchronously (a few ms) so /feedback can always find the id;
        # a logging failure never fails the prediction itself.
        if settings.api.log_predictions:
            store = _get_store(app)
            try:
                if store is None:
                    raise RuntimeError("prediction store unavailable")
                store.log_predictions(records)
            except Exception as exc:
                metrics.PREDICTION_LOG_ERRORS.inc()
                logger.error("prediction_logging_failed", error=str(exc), n=len(records))
        return responses, loaded

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Liveness: always 200; ``status`` is ``degraded`` until a model is loaded."""
        current = app.state.model_service.current
        return HealthResponse(
            status="ok" if current else "degraded",
            model_loaded=current is not None,
            model_version=current.version if current else None,
            uptime_seconds=round(time.monotonic() - app.state.started_at, 1),
        )

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
    def model_info() -> ModelInfoResponse:
        """Which registry version is being served, and how it was trained."""
        current = app.state.model_service.current
        if current is None:
            raise HTTPException(
                status_code=503, detail=app.state.model_service.last_error or "No model"
            )
        return ModelInfoResponse(
            model_name=current.name,
            model_version=current.version,
            model_uri=current.model_uri,
            source=current.source,
            alias=current.alias,
            run_id=current.run_id,
            model_type=current.model_type,
            loaded_at=current.loaded_at,
            decision_threshold=settings.training.decision_threshold,
            features=FEATURE_COLUMNS,
            tags=current.tags,
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(patient: PatientFeatures) -> PredictionResponse:
        """Predict heart disease risk for one patient."""
        responses, _ = run_predictions([patient])
        return responses[0]

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
    def predict_batch(body: BatchPredictionRequest) -> BatchPredictionResponse:
        """Predict for up to ``api.max_batch_size`` patients in one call."""
        if len(body.instances) > settings.api.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"Batch size {len(body.instances)} exceeds {settings.api.max_batch_size}",
            )
        responses, loaded = run_predictions(body.instances)
        return BatchPredictionResponse(
            predictions=responses, model_name=loaded.name, model_version=loaded.version
        )

    @app.post("/feedback", response_model=FeedbackResponse, tags=["monitoring"])
    def feedback(body: FeedbackRequest) -> FeedbackResponse:
        """Attach confirmed diagnoses to earlier predictions (used for performance monitoring)."""
        store = _get_store(app)
        if store is None:
            raise HTTPException(status_code=503, detail="Prediction store unavailable")
        updated, unknown = store.add_feedback([(i.request_id, i.actual) for i in body.items])
        for item in body.items:
            metrics.FEEDBACK_RECEIVED.labels(str(item.actual)).inc()
        return FeedbackResponse(updated=updated, unknown_request_ids=unknown)

    @app.post("/model/reload", tags=["ops"])
    async def reload_model() -> dict[str, object]:
        """Re-resolve the champion alias now instead of waiting for the next poll."""
        changed = await asyncio.to_thread(_refresh_model, app.state.model_service)
        current = app.state.model_service.current
        return {
            "reloaded": changed,
            "model_version": current.version if current else None,
            "error": app.state.model_service.last_error,
        }

    @app.get("/metrics", tags=["ops"])
    def prometheus_metrics() -> Response:
        """Prometheus exposition format."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
