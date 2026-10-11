# Screenshots

README images, captured from the running docker-compose stack with
[`scripts/capture_screenshots.py`](../../scripts/capture_screenshots.py):

```bash
pip install -r requirements-docs.txt && python -m playwright install chromium
# after the drifted batch, before retraining (drift visible):
python -m scripts.capture_screenshots --only monitor drift performance api
# after `src.retrain` promoted a challenger (registry shows both aliases):
python -m scripts.capture_screenshots --only mlflow
```

| File | Page |
|---|---|
| `monitor-dashboard.png` | http://localhost:8050 |
| `drift-report.png` | http://localhost:8050/reports/drift |
| `performance-report.png` | http://localhost:8050/reports/performance |
| `mlflow-runs.png` | http://localhost:5000 (experiment runs) |
| `mlflow-registry.png` | http://localhost:5000 (registered model) |
| `api-docs.png` | http://localhost:8000/docs |
