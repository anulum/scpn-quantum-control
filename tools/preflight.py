# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
#!/usr/bin/env python3
"""Local CI preflight — mirrors every CI gate so failures are caught before push.

Gates (in order):
  1. ruff check      — lint errors
  2. ruff format     — formatting drift
  3. version-sync    — version string consistency across 5 carrier files
  4. mypy            — type errors
  5. pytest+coverage — tests + coverage threshold (--cov-fail-under=92, CI=95)
  6. bandit          — security scan

Usage:
  python tools/preflight.py                # all gates (default)
  python tools/preflight.py --no-tests     # skip pytest entirely (quick lint pass)
  python tools/preflight.py --no-coverage  # run tests without coverage threshold
"""

from __future__ import annotations

import subprocess  # noqa: S404
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_PY = sys.executable

_PYTEST_BASE = [
    _PY,
    "-m",
    "pytest",
    "tests/",
    "-x",
    "--tb=short",
    "-q",
    "--ignore=tests/test_hardware_runner.py",
]

_PYTEST_COV = _PYTEST_BASE + [
    "--cov=scpn_quantum_control",
    "--cov-fail-under=92",  # CI enforces 95 on Linux; 3-pt buffer for Windows/Linux branch delta
]

STATIC_GATES: list[tuple[str, list[str]]] = [
    ("ruff check", [_PY, "-m", "ruff", "check", "src/", "tests/"]),
    ("ruff format", [_PY, "-m", "ruff", "format", "--check", "src/", "tests/"]),
    ("version-sync", [_PY, "scripts/check_version_consistency.py"]),
    ("mypy", [_PY, "-m", "mypy"]),
]

BANDIT_GATE: tuple[str, list[str]] = (
    "bandit",
    [_PY, "-m", "bandit", "-r", "src/", "-ll", "-q"],
)


def run_gate(name: str, cmd: list[str]) -> bool:
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)  # noqa: S603
    elapsed = time.monotonic() - t0
    if result.returncode == 0:
        print(f"  PASS  {name} ({elapsed:.1f}s)")
        return True
    print(f"  FAIL  {name} ({elapsed:.1f}s)")
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines()[-10:]:
            print(f"        {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines()[-10:]:
            print(f"        {line}")
    return False


def main() -> int:
    skip_tests = "--no-tests" in sys.argv
    no_coverage = "--no-coverage" in sys.argv

    gates: list[tuple[str, list[str]]] = list(STATIC_GATES)

    if not skip_tests:
        if no_coverage:
            gates.append(("pytest", _PYTEST_BASE))
        else:
            gates.append(("pytest + coverage", _PYTEST_COV))

    gates.append(BANDIT_GATE)

    print(f"preflight: {len(gates)} gates")
    print()

    t_start = time.monotonic()
    failed: list[str] = []

    for name, cmd in gates:
        if not run_gate(name, cmd):
            failed.append(name)
            break

    elapsed = time.monotonic() - t_start
    print()
    if failed:
        print(f"BLOCKED: {', '.join(failed)} ({elapsed:.1f}s)")
        return 1
    print(f"ALL CLEAR: ready to push ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
