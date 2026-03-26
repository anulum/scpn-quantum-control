# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

.PHONY: test lint fmt check docs preflight clean install

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=scpn_quantum_control --cov-report=term-missing --cov-fail-under=95

lint:
	ruff check src/ tests/ examples/

fmt:
	ruff format src/ tests/ examples/
	ruff check --fix src/ tests/ examples/

check:
	ruff check src/ tests/ examples/
	ruff format --check src/ tests/ examples/
	mypy

docs:
	mkdocs build -d site

docs-serve:
	mkdocs serve

preflight:
	ruff check src/ tests/ examples/
	ruff format --check src/ tests/ examples/
	mypy
	bandit -r src/ scripts/ -ll -q
	pytest tests/ -v --tb=short -x --ignore=tests/test_hardware_runner.py

preflight-quick:
	ruff check src/ tests/ examples/
	ruff format --check src/ tests/ examples/
	mypy

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache dist build site *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
