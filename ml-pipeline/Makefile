# Heart Disease MLOps - common tasks.  Run `make help` for the list.
#
# Pipeline jobs run in the local virtualenv by default. Add DOCKER=1 to run
# them inside the `trainer` container instead (no local Python needed):
#   make train DOCKER=1

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3
VENV ?= .venv
ifeq ($(OS),Windows_NT)
  VENV_BIN := $(VENV)/Scripts
else
  VENV_BIN := $(VENV)/bin
endif
PY := $(VENV_BIN)/python
COMPOSE ?= docker compose
SERVE_PORT ?= 8001

ifdef DOCKER
  RUN := $(COMPOSE) run --rm trainer python
  API_URL ?= http://api:8000
else
  RUN := $(PY)
  API_URL ?= http://localhost:8000
endif

.PHONY: help setup ingest train serve drift-demo retrain hf-space test test-fast lint format up down logs clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv, install pinned deps, create .env, install pre-commit hooks
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt
	@test -f .env || cp .env.example .env
	-$(VENV_BIN)/pre-commit install

ingest: ## Download, validate and split the UCI dataset into data/
	$(RUN) -m src.ingest

train: ingest ## Train candidates, log to MLflow, register + promote the champion
	$(RUN) -m src.train

serve: ## Run the API locally with auto-reload on SERVE_PORT (default 8001)
	$(PY) -m uvicorn api.main:app --reload --port $(SERVE_PORT)

drift-demo: ## Send normal + drifted (labeled) traffic to the API, then check drift
	$(RUN) -m scripts.send_sample_requests --api-url $(API_URL) --batch data/production/batch_01.parquet
	$(RUN) -m scripts.simulate_drift
	$(RUN) -m scripts.send_sample_requests --api-url $(API_URL) --batch data/production/drifted_batch.parquet
	$(RUN) -m src.drift

retrain: ## Drift/performance check -> retrain -> promote only if better
	$(RUN) -m src.retrain

hf-space: ## Export the champion into a Hugging Face Docker Space build context (hf_space/build)
	$(RUN) -m scripts.export_model --output hf_space/build

test: ## Run the full test suite with coverage
	$(PY) -m pytest --cov --cov-report=term-missing:skip-covered

test-fast: ## Run tests that do not train models
	$(PY) -m pytest -m "not slow"

lint: ## Ruff lint + format check
	$(VENV_BIN)/ruff check .
	$(VENV_BIN)/ruff format --check .

format: ## Auto-fix lint issues and format code
	$(VENV_BIN)/ruff check --fix .
	$(VENV_BIN)/ruff format .

up: ## Build and start the docker-compose stack
	$(COMPOSE) up -d --build

down: ## Stop the stack (keeps volumes)
	$(COMPOSE) down

logs: ## Follow API / MLflow / monitor logs
	$(COMPOSE) logs -f api mlflow monitor

clean: ## Stop the stack AND delete volumes, local stores and generated data
	$(COMPOSE) down -v
	rm -rf mlruns mlartifacts reports *.db data/raw data/processed data/production data/*.db
