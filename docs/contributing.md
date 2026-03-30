# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Contributing

# Contributing

## Setup

```bash
git clone https://github.com/anulum/scpn-quantum-control.git
cd scpn-quantum-control
pip install -e ".[dev]"
pre-commit install
pytest tests/ -x -q
```

## Code Quality Gates

All contributions must pass the full CI pipeline:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
pytest tests/ -v --ignore=tests/test_hardware_runner.py
```

## Testing Requirements

Every new module needs a corresponding `tests/test_<module>.py` with:

- At least one physics verification (quantum result matches classical reference)
- At least one circuit validity check (transpiles on AerSimulator without error)
- At least one edge case (zero input, identity coupling, etc.)

Statistical tests should use `n_shots >= 1000` for any assertion on measurement
outcomes.

## Hardware Experiments

Hardware runs require IBM Quantum credentials and available QPU budget. Save
results as JSON in `results/` with IBM job ID for reproducibility.

## Pull Request Checklist

- [ ] Tests pass on Python 3.10–3.13
- [ ] Lint and type-check clean
- [ ] No new dependencies without justification
- [ ] CHANGELOG.md updated for user-facing changes
- [ ] Hardware experiments documented with JSON + job ID

## Contact

Questions or proposals: [protoscience@anulum.li](mailto:protoscience@anulum.li) |
[GitHub Discussions](https://github.com/anulum/scpn-quantum-control/discussions)
