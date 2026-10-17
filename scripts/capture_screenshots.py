"""Capture the README screenshots from the running docker-compose stack.

Requires the optional docs dependencies::

    pip install -r requirements-docs.txt && python -m playwright install chromium

Usage::

    python -m scripts.capture_screenshots                     # all pages
    python -m scripts.capture_screenshots --only monitor drift

Take ``monitor`` / ``drift`` while drift is present (after the drifted batch,
before retraining) and ``mlflow-*`` after a promotion, so the registry shows
the ``champion`` and ``previous-champion`` aliases.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import PROJECT_ROOT, get_settings

if TYPE_CHECKING:
    from playwright.sync_api import Page


@dataclass(frozen=True)
class Shot:
    name: str
    url: str
    wait_ms: int = 2500
    full_page: bool = False
    mlflow_ui: bool = False  # close the assistant panel, expand nested runs


def _prepare_mlflow(page: Page) -> None:
    """Tidy the MLflow 3 UI: close the assistant side panel, expand child runs."""
    page.keyboard.press("Escape")
    close = page.get_by_role("button", name="Close")
    if close.count():
        close.first.click()
    expanders = page.locator('button[data-component-id$="runnamecellrenderer.tsx_46"]')
    for i in range(expanders.count()):
        expanders.nth(i).click()
        page.wait_for_timeout(500)
    page.wait_for_timeout(1000)


def _shots(api: str, monitor: str, mlflow_ui: str, experiment_id: str, model: str) -> list[Shot]:
    return [
        Shot("monitor-dashboard", f"{monitor}/?refresh=true", 1500, full_page=True),
        Shot("drift-report", f"{monitor}/reports/drift", 6000),
        Shot("performance-report", f"{monitor}/reports/performance", 6000),
        Shot("api-docs", f"{api}/docs", 2500, full_page=True),
        Shot(
            "mlflow-runs", f"{mlflow_ui}/#/experiments/{experiment_id}/runs", 6000, mlflow_ui=True
        ),
        Shot("mlflow-registry", f"{mlflow_ui}/#/models/{model}", 5000, mlflow_ui=True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture README screenshots.")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--monitor", default="http://localhost:8050")
    parser.add_argument("--mlflow", default="http://localhost:5000")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "images")
    parser.add_argument("--only", nargs="*", help="name prefixes, e.g. monitor drift mlflow")
    args = parser.parse_args()

    from mlflow import MlflowClient
    from playwright.sync_api import sync_playwright

    settings = get_settings()
    experiment = MlflowClient(tracking_uri=args.mlflow).get_experiment_by_name(
        settings.mlflow.experiment_name
    )
    shots = _shots(
        args.api,
        args.monitor,
        args.mlflow,
        experiment.experiment_id if experiment else "0",
        settings.mlflow.registered_model_name,
    )
    if args.only:
        shots = [s for s in shots if any(s.name.startswith(p) for p in args.only)]

    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        for shot in shots:
            # "load" + a fixed delay: the MLflow UI polls forever, so it never goes idle.
            page.goto(shot.url, wait_until="load", timeout=60_000)
            page.wait_for_timeout(shot.wait_ms)
            if shot.mlflow_ui:
                _prepare_mlflow(page)
            path = args.out / f"{shot.name}.png"
            page.screenshot(path=str(path), full_page=shot.full_page)
            print(f"saved {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size // 1024} KB)")
        browser.close()


if __name__ == "__main__":
    main()
