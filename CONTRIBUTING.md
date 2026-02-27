# Contributing to scpn-quantum-control

## Setup

```bash
pip install -e ".[dev]"
```

## Tests

```bash
pytest tests/ -v
```

## Code standards

- Follow the anti-slop policy in the root CLAUDE.md
- Every magic number references its source (paper equation, calibration table)
- Quantum circuits must transpile on AerSimulator without error
- Statistical tests compare quantum vs classical outputs (not exact match)
