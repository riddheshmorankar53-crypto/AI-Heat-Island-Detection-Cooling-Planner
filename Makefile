.PHONY: help setup data train api web test lint format docker-up docker-down

PYTHON ?= python

help:
	@echo "Urban Cooling Planner - Available Commands:"
	@echo "  make setup        Install Python and Frontend dependencies"
	@echo "  make data         Generate 100m grid and pedestrian street graph"
	@echo "  make train        Train ML models (Ridge, LightGBM, Spatial GNN)"
	@echo "  make api          Launch FastAPI backend server on port 8000"
	@echo "  make web          Launch React Vite frontend on port 5173"
	@echo "  make test         Execute test suite (unit, integration & smoke tests)"
	@echo "  make lint         Run ruff linter"
	@echo "  make format       Run ruff formatter"
	@echo "  make docker-up    Start full stack in Docker Compose"
	@echo "  make docker-down  Stop Docker Compose stack"

setup:
	$(PYTHON) -m pip install -e ".[dev]"
	cd frontend && npm install

data:
	$(PYTHON) data/pipeline.py

train:
	$(PYTHON) ml/train.py

api:
	$(PYTHON) -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

web:
	cd frontend && npm run dev

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
