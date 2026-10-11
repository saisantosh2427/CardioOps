"""Prediction log store (SQLAlchemy; SQLite locally, Postgres in docker-compose).

The API writes every served prediction here; ground-truth labels arrive later
through ``/feedback``. Monitoring and retraining read *this* table, so drift and
performance are measured on real served traffic rather than offline files.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    func,
    select,
    update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.columns import FEATURE_COLUMNS, TARGET


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    """One served prediction (and, once known, its ground-truth label)."""

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(32), index=True)
    features: Mapped[dict[str, Any]] = mapped_column(JSON)
    prediction: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feedback_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


@dataclass(frozen=True)
class PredictionRecord:
    """Data the API hands to the store for one prediction."""

    request_id: str
    model_name: str
    model_version: str
    features: dict[str, Any]
    prediction: int
    probability: float
    latency_ms: float | None = None


class PredictionStore:
    """Thin repository around the ``prediction_logs`` table."""

    def __init__(self, database_url: str) -> None:
        connect_args: dict[str, Any] = {}
        if database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            db_file = database_url.split("sqlite:///", 1)[-1]
            if db_file and db_file != ":memory:":
                Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
        Base.metadata.create_all(self.engine)
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    def log_predictions(self, records: Iterable[PredictionRecord]) -> int:
        """Insert prediction records; returns the number written."""
        now = datetime.now(UTC)
        rows = [
            PredictionLog(
                request_id=r.request_id,
                created_at=now,
                model_name=r.model_name,
                model_version=r.model_version,
                features=r.features,
                prediction=r.prediction,
                probability=r.probability,
                latency_ms=r.latency_ms,
            )
            for r in records
        ]
        with self._session.begin() as session:
            session.add_all(rows)
        return len(rows)

    def add_feedback(self, labels: Sequence[tuple[str, int]]) -> tuple[int, list[str]]:
        """Attach ground-truth labels. Returns (rows updated, unknown request ids)."""
        updated, missing = 0, []
        now = datetime.now(UTC)
        with self._session.begin() as session:
            for request_id, actual in labels:
                result = session.execute(
                    update(PredictionLog)
                    .where(PredictionLog.request_id == request_id)
                    .values(actual=int(actual), feedback_at=now)
                )
                if result.rowcount:
                    updated += result.rowcount
                else:
                    missing.append(request_id)
        return updated, missing

    def load_recent(self, limit: int, labeled_only: bool = False) -> pd.DataFrame:
        """Most recent ``limit`` predictions (oldest first) as a flat DataFrame.

        Columns: the model features, ``prediction``, ``probability``,
        ``target`` (ground truth, NaN if unknown), ``model_version``,
        ``request_id`` and ``created_at``.
        """
        stmt = select(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit)
        if labeled_only:
            stmt = stmt.where(PredictionLog.actual.is_not(None))
        with self._session() as session:
            rows = list(reversed(session.scalars(stmt).all()))
        columns = [
            *FEATURE_COLUMNS,
            "prediction",
            "probability",
            TARGET,
            "model_version",
            "request_id",
            "created_at",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        records = [
            {
                **{col: row.features.get(col) for col in FEATURE_COLUMNS},
                "prediction": row.prediction,
                "probability": row.probability,
                TARGET: row.actual,
                "model_version": row.model_version,
                "request_id": row.request_id,
                "created_at": row.created_at,
            }
            for row in rows
        ]
        frame = pd.DataFrame.from_records(records, columns=columns)
        frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
        return frame

    def counts(self) -> dict[str, int]:
        """Total and labeled prediction counts."""
        with self._session() as session:
            total = session.scalar(select(func.count(PredictionLog.id))) or 0
            labeled = (
                session.scalar(
                    select(func.count(PredictionLog.id)).where(PredictionLog.actual.is_not(None))
                )
                or 0
            )
        return {"total": int(total), "labeled": int(labeled)}

    def dispose(self) -> None:
        self.engine.dispose()
