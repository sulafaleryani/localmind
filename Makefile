-include .env
OLLAMA_MODEL ?= llama3.2
PY := .venv/bin

.PHONY: setup run pull-model run-docker test lint format clean

setup:
	python3 -m venv .venv && $(PY)/pip install -r requirements.txt
	@[ -f .env ] || cp .env.example .env

run:
	$(PY)/streamlit run app.py

pull-model:
	ollama pull $(OLLAMA_MODEL)

run-docker:
	docker compose up --build

test:
	$(PY)/pytest -q

lint:
	$(PY)/ruff check . && $(PY)/black --check .

format:
	$(PY)/ruff check --fix . && $(PY)/black . && $(PY)/isort .

clean:
	rm -rf .chroma .venv .pytest_cache .ruff_cache && find . -name __pycache__ -type d -prune -exec rm -rf {} +
