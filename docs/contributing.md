# Contributing

See [CONTRIBUTING.md](https://github.com/anulum/scpn-quantum-control/blob/main/CONTRIBUTING.md) for the full contribution guide.

## Quick setup

```bash
git clone https://github.com/anulum/scpn-quantum-control.git
cd scpn-quantum-control
pip install -e ".[dev]"
pytest tests/ -x -q
```

## Code quality gates

All PRs must pass:

```bash
ruff check src/ tests/       # zero errors
ruff format --check src/ tests/  # zero diffs
python -m mypy src/           # zero errors
pytest tests/ -x -q           # 456 tests pass
```

## Anti-slop policy

This project enforces the anti-slop code policy from `CLAUDE.md`.
No narration comments, no buzzword naming, no placeholder values
without tracked issues, no trivial wrappers.
