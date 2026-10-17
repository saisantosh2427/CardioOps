---
title: Heart Disease Risk API
emoji: 🫀
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: FastAPI heart-disease risk model from an MLOps pipeline
---

# Heart Disease Risk API

FastAPI service exported from the
[heart-disease MLOps pipeline](https://github.com/<your-user>/ml-pipeline).
The registry champion is baked into the image (`model/`), so the Space needs no
MLflow server.

- Interactive docs: `/docs`
- `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/batch`, `GET /metrics`

```bash
curl -X POST https://<your-user>-<space-name>.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":4,"trestbps":145,"chol":233,"fbs":1,"restecg":2,
       "thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

**Not a medical device.** Educational MLOps demo trained on the public UCI
Heart Disease dataset (CC BY 4.0).
