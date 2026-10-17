"""Monitoring dashboard: Evidently reports over the traffic the API actually served.

- ``GET /``                     summary dashboard (drift, performance, retrain verdict)
- ``GET /reports/drift``        Evidently data-drift report (reference vs. served window)
- ``GET /reports/performance``  Evidently classification report on labeled traffic
- ``GET /api/summary``          the same summary as JSON
- ``GET /metrics``              Prometheus gauges (drift share, retrain flag, ...)
- ``GET /health``

Reports are cached for ``monitoring.report_cache_seconds``; add ``?refresh=true``
to recompute. Run: ``uvicorn monitor.app:app --port 8050``
"""

from __future__ import annotations

import html
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest

from src import __version__
from src.columns import TARGET
from src.config import Settings, get_settings
from src.drift import (
    baseline_metrics_from_tags,
    check_performance,
    decide_retrain,
    load_reference,
    run_drift_report,
    run_performance_report,
)
from src.logging_utils import configure_logging, get_logger
from src.store import PredictionStore

logger = get_logger("monitor")

DRIFT_REPORT = "monitor_drift_report.html"
PERFORMANCE_REPORT = "monitor_performance_report.html"


class MonitorService:
    """Computes (and caches) the monitoring summary and Evidently reports."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.reports_dir = settings.reports_dir
        self._lock = threading.Lock()
        self._summary: dict[str, Any] | None = None
        self._computed_at = 0.0
        self._store: PredictionStore | None = None

        self.registry = CollectorRegistry()
        self.g_drift_share = Gauge(
            "monitor_drift_share", "Share of features drifted", registry=self.registry
        )
        self.g_dataset_drift = Gauge(
            "monitor_dataset_drift", "1 if dataset drift detected", registry=self.registry
        )
        self.g_window = Gauge(
            "monitor_window_predictions", "Predictions in the window", registry=self.registry
        )
        self.g_labeled = Gauge(
            "monitor_labeled_predictions",
            "Labeled predictions of the champion in the window",
            registry=self.registry,
        )
        self.g_metric = Gauge(
            "monitor_champion_metric",
            "Champion metric on labeled traffic",
            ["metric"],
            registry=self.registry,
        )
        self.g_retrain = Gauge(
            "monitor_retrain_recommended", "1 if retraining is recommended", registry=self.registry
        )

    @property
    def store(self) -> PredictionStore:
        if self._store is None:
            self._store = PredictionStore(self.settings.database_url)
        return self._store

    def summary(self, refresh: bool = False) -> dict[str, Any]:
        with self._lock:
            fresh = (
                time.monotonic() - self._computed_at < self.settings.monitoring.report_cache_seconds
            )
            if self._summary is not None and fresh and not refresh:
                return self._summary
            try:
                self._summary = self._compute()
            except Exception as exc:
                logger.exception("monitoring_failed")
                self._summary = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            self._computed_at = time.monotonic()
            self._export_metrics(self._summary)
            return self._summary

    def _compute(self) -> dict[str, Any]:
        settings = self.settings
        reference, champion = load_reference(settings)
        current = self.store.load_recent(settings.monitoring.window_size)
        counts = self.store.counts()
        summary: dict[str, Any] = {
            "status": "ok",
            "computed_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "champion_version": str(champion.version) if champion else None,
            "champion_model_type": (champion.tags or {}).get("model_type") if champion else None,
            "total_predictions": counts["total"],
            "total_labeled": counts["labeled"],
            "window_size": len(current),
            "drift": None,
            "performance": None,
            "decision": None,
            "reports": {"drift": False, "performance": False},
        }
        if current.empty:
            summary["status"] = "no_data"
            return summary

        drift = run_drift_report(
            reference, current, settings, html_path=self.reports_dir / DRIFT_REPORT
        )
        summary["drift"] = asdict(drift)
        summary["reports"]["drift"] = True

        # Performance of the *current* champion only: rows served by older
        # versions say nothing about the model in production now.
        labeled = current[current[TARGET].notna()]
        if champion is not None:
            labeled = labeled[labeled["model_version"] == str(champion.version)]
        performance = check_performance(
            labeled[TARGET].astype(int),
            labeled["probability"],
            baseline_metrics_from_tags(dict(champion.tags or {})) if champion else None,
            settings,
        )
        summary["performance"] = asdict(performance)
        if performance.current is not None:
            run_performance_report(labeled, self.reports_dir / PERFORMANCE_REPORT)
            summary["reports"]["performance"] = True
        summary["decision"] = asdict(decide_retrain(drift, performance, settings))
        return summary

    def _export_metrics(self, summary: dict[str, Any]) -> None:
        drift = summary.get("drift") or {}
        perf = summary.get("performance") or {}
        self.g_drift_share.set(drift.get("drift_share", 0.0))
        self.g_dataset_drift.set(float(drift.get("dataset_drift", False)))
        self.g_window.set(summary.get("window_size", 0))
        self.g_labeled.set(perf.get("n_labeled", 0))
        for name, value in (perf.get("current") or {}).items():
            self.g_metric.labels(metric=name).set(value)
        self.g_retrain.set(float((summary.get("decision") or {}).get("retrain", False)))


_CSS = """
body { font-family: system-ui, sans-serif; margin: 0 auto; max-width: 1000px;
       padding: 24px 16px; color: #1f2933; }
nav a { margin-right: 14px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
         gap: 12px; margin: 20px 0; }
.card { border: 1px solid #d9e2ec; border-radius: 8px; padding: 12px; }
.card.good { border-left: 4px solid #2f9e44; }
.card.bad { border-left: 4px solid #e03131; }
.label, .note { font-size: 12px; color: #52606d; }
.value { font-size: 26px; font-weight: 600; margin: 4px 0; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
td, th { border-bottom: 1px solid #e4e7eb; padding: 6px; text-align: left; }
tr.bad td, p.bad { color: #c92a2a; font-weight: 600; }
code { background: #f0f4f8; padding: 1px 4px; }
"""


def _card(title: str, value: str, note: str = "", tone: str = "") -> str:
    return (
        f'<div class="card {tone}"><div class="label">{html.escape(title)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="note">{html.escape(note)}</div></div>'
    )


def render_dashboard(summary: dict[str, Any], settings: Settings) -> str:
    """Self-contained HTML dashboard (no external assets)."""
    links = (
        f'<a href="/reports/drift">Drift report</a>'
        f'<a href="/reports/performance">Performance report</a>'
        f'<a href="/api/summary">JSON</a>'
        f'<a href="{html.escape(settings.monitoring.mlflow_ui_url)}">MLflow UI</a>'
        f'<a href="{html.escape(settings.monitoring.api_url)}/docs">API docs</a>'
        f'<a href="/?refresh=true">Refresh now</a>'
    )
    status = summary.get("status")
    if status == "error":
        body = f'<p class="bad">Monitoring failed: {html.escape(summary["error"])}</p>'
    elif status == "no_data":
        body = (
            "<p>No predictions logged yet. Send traffic with "
            "<code>make drift-demo</code> or <code>POST /predict</code>.</p>"
        )
    else:
        drift, perf, decision = summary["drift"], summary["performance"], summary["decision"]
        metric = perf["metric"]
        current_metric = (perf.get("current") or {}).get(metric)
        cards = [
            _card(
                "Champion",
                f"v{summary['champion_version']}" if summary["champion_version"] else "none",
                summary.get("champion_model_type") or "",
            ),
            _card(
                "Predictions",
                f"{summary['total_predictions']}",
                f"{summary['total_labeled']} labeled; window {summary['window_size']}",
            ),
            _card(
                "Drifted features",
                f"{drift['n_drifted']}/{drift['n_columns']}",
                f"share {drift['drift_share']:.2f} (threshold "
                f"{settings.monitoring.drift_share_threshold})",
                "bad" if drift["dataset_drift"] else "good",
            ),
            _card(
                f"Champion {metric} (live)",
                f"{current_metric:.3f}" if current_metric is not None else "n/a",
                f"baseline {perf['baseline_value']:.3f}"
                if perf.get("baseline_value") is not None
                else f"needs {settings.monitoring.min_labeled_samples} labels",
                "bad" if perf["degraded"] else "good",
            ),
            _card(
                "Retrain?",
                "YES" if decision["retrain"] else "no",
                "run `make retrain`" if decision["retrain"] else "",
                "bad" if decision["retrain"] else "good",
            ),
        ]
        rows = "".join(
            f"<tr class={'bad' if t['drifted'] else ''}><td>{html.escape(col)}</td>"
            f"<td>{html.escape(t['method'])}</td><td>{t['score']:.4f}</td>"
            f"<td>{'drift' if t['drifted'] else 'ok'}</td></tr>"
            for col, t in sorted(drift["column_tests"].items())
        )
        reasons = "".join(f"<li>{html.escape(r)}</li>" for r in decision["reasons"])
        body = (
            f'<div class="cards">{"".join(cards)}</div>'
            f"<h2>Decision</h2><ul>{reasons}</ul>"
            "<h2>Per-feature drift tests (p-values)</h2>"
            '<div class="scroll"><table><tr><th>feature</th><th>test</th>'
            f"<th>p-value</th><th>result</th></tr>{rows}</table></div>"
        )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Heart Disease Model Monitor</title><style>{_CSS}</style></head>
<body><h1>Heart Disease Model Monitor</h1><nav>{links}</nav>
<p class="note">Computed {html.escape(str(summary.get("computed_at", "")))} &middot;
reference = champion training data &middot; current = last {settings.monitoring.window_size}
served predictions</p>{body}</body></html>"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.project.log_level, settings.project.log_format)
    service = MonitorService(settings)
    app = FastAPI(title="Heart Disease Model Monitor", version=__version__)
    app.state.service = service

    @app.get("/", response_class=HTMLResponse)
    def dashboard(refresh: bool = Query(False)) -> str:
        return render_dashboard(service.summary(refresh=refresh), settings)

    @app.get("/api/summary")
    def summary(refresh: bool = Query(False)) -> dict[str, Any]:
        return service.summary(refresh=refresh)

    def _report(name: str, key: str) -> FileResponse:
        result = service.summary()
        path: Path = service.reports_dir / name
        if not (result.get("reports") or {}).get(key) or not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No {key} report yet (status: {result.get('status')}); "
                "it needs logged predictions"
                + (" with enough ground-truth labels" if key == "performance" else ""),
            )
        return FileResponse(path, media_type="text/html")

    @app.get("/reports/drift")
    def drift_report() -> FileResponse:
        return _report(DRIFT_REPORT, "drift")

    @app.get("/reports/performance")
    def performance_report() -> FileResponse:
        return _report(PERFORMANCE_REPORT, "performance")

    @app.get("/metrics")
    def metrics() -> Response:
        service.summary()
        return Response(generate_latest(service.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
