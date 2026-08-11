PYTHON_PROJECT := Python
DATA_CONFIG := configs/data/open_images_smoke.json

.PHONY: data-setup data-check data-plan data-smoke

data-setup:
	uv sync --project $(PYTHON_PROJECT)

data-check: data-setup
	uv run --project $(PYTHON_PROJECT) ruff format --check $(PYTHON_PROJECT)/data_pipeline $(PYTHON_PROJECT)/tests
	uv run --project $(PYTHON_PROJECT) ruff check $(PYTHON_PROJECT)/data_pipeline $(PYTHON_PROJECT)/tests
	uv run --project $(PYTHON_PROJECT) pytest
	uv run --project $(PYTHON_PROJECT) python -m data_pipeline check
	uv run --project $(PYTHON_PROJECT) python -m data_pipeline acquire --config $(DATA_CONFIG) --dry-run

data-plan: data-setup
	uv run --project $(PYTHON_PROJECT) python -m data_pipeline acquire --config $(DATA_CONFIG) --dry-run

data-smoke: data-setup
	uv run --project $(PYTHON_PROJECT) python -m data_pipeline acquire --config $(DATA_CONFIG)
