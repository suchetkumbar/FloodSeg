.PHONY: install lint format typecheck test data-pipeline stats baselines train grid aggregate clean

PYTHON = ./.venv/Scripts/python.exe
PYTEST = ./.venv/Scripts/pytest.exe
RUFF = ./.venv/Scripts/ruff.exe
MYPY = ./.venv/Scripts/mypy.exe

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(RUFF) check src tests scripts

format:
	$(RUFF) format src tests scripts

typecheck:
	$(MYPY) src tests

test:
	$(PYTEST) tests -v

data-pipeline:
	$(PYTHON) scripts/run_milestone1_pipeline.py

stats:
	$(PYTHON) -m floodseg.cli compute-stats --split all

baselines:
	$(PYTHON) scripts/run_baselines.py

train:
	$(PYTHON) scripts/train_model.py

grid:
	$(PYTHON) scripts/run_m3_grid.py

aggregate:
	$(PYTHON) scripts/aggregate_m3.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
